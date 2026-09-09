"""Exercise the module's saved-preset boundary without a live user Library."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier
from types import SimpleNamespace
import unittest

from test_captioner_export import captioner


class MemoryLibraryDB:
    def __init__(self):
        self.cards = {}
        self.next_id = 1

    def create_card(self, data):
        card_id = self.next_id
        self.next_id += 1
        self.cards[card_id] = {**deepcopy(data), 'id': card_id, 'updated_at': card_id}
        return card_id

    def get_card(self, card_id):
        return deepcopy(self.cards.get(card_id))

    def update_card(self, card_id, data):
        self.cards[card_id].update(deepcopy(data))

    def delete_card(self, card_id):
        return self.cards.pop(card_id).get('attachment')

    def list_cards(self, *, type_filter=None, tag_filter=None, limit=200, offset=0):
        cards = [deepcopy(card) for card in self.cards.values()
                 if (not type_filter or card.get('type') == type_filter)
                 and (not tag_filter or tag_filter in card.get('tags', []))]
        return {'cards': cards[offset:offset+limit], 'total': len(cards)}


class Handler:
    def __init__(self, data):
        self.data = data
        self.status = 200

    def read_body_json(self, _length):
        return self.data

    def respond_json(self, data, status=200):
        self.response, self.status = data, status


class PresetTests(unittest.TestCase):
    def setUp(self):
        self.db = MemoryLibraryDB()
        self.cleaned = []
        self.library = SimpleNamespace(db=self.db, _cleanup_attachment_if_unused=self.cleaned.append)
        self.module = captioner.CaptionerModule()
        self.module.hub = SimpleNamespace(registry=SimpleNamespace(get=lambda key: self.library if key == 'library' else None))

    def call(self, method, data):
        handler = Handler(data)
        getattr(self.module, method)(handler, 0, 'application/json')
        return handler

    def seed(self, title, **extra):
        return self.db.create_card({'title': title, 'content': 'Original', 'type': 'captioner', **extra})

    def test_new_preset_has_captioner_type_and_tag(self):
        result = self.call('_save_preset', {'title': '  Portrait  ', 'content': 'Text with {TRIGGER}'})
        self.assertEqual(result.status, 200)
        self.assertEqual(result.response['preset']['title'], 'Portrait')
        self.assertEqual(result.response['preset']['type'], 'captioner')
        self.assertEqual(result.response['preset']['tags'], ['captioner-preset'])

    def test_duplicate_names_ignore_case_whitespace_and_unicode_compatibility(self):
        self.seed('Café Portrait')
        for title in ['Café Portrait', '  CAFÉ   Portrait ', 'Cafe\u0301\tPortrait', 'Ｃａｆé Portrait']:
            result = self.call('_save_preset', {'title': title, 'content': 'Different prompt'})
            self.assertEqual(result.status, 409, title)
        self.assertEqual(len(self.db.cards), 1)

    def test_two_simultaneous_saves_create_one_preset(self):
        barrier = Barrier(2)
        def save(_):
            barrier.wait(timeout=2)
            return self.call('_save_preset', {'title': 'Concurrent', 'content': 'Text'}).status
        with ThreadPoolExecutor(max_workers=2) as pool:
            statuses = list(pool.map(save, range(2)))
        self.assertEqual(sorted(statuses), [200, 409])
        self.assertEqual(len(self.db.cards), 1)

    def test_duplicate_check_includes_later_pages_and_legacy_tags(self):
        for i in range(205):
            self.seed(f'Preset {i}')
        legacy_id = self.seed('Legacy shared', type='system', tags=['captioner-preset'])
        self.assertEqual(self.call('_save_preset', {'title': 'Preset 204', 'content': 'New'}).status, 409)
        result = self.call('_save_preset', {'title': 'LEGACY SHARED', 'content': 'New'})
        self.assertEqual(result.status, 409)
        self.assertEqual(result.response['duplicate_id'], legacy_id)
        handler = Handler(None)
        self.module._get_presets(handler, {})
        self.assertEqual(len(handler.response['cards']), 206)

    def test_update_own_title_is_allowed_but_renaming_to_another_is_not(self):
        a, b = self.seed('First'), self.seed('Second')
        updated = self.call('_save_preset', {'id': a, 'title': ' FIRST ', 'content': 'Edited'})
        self.assertEqual(updated.status, 200)
        self.assertEqual(self.db.get_card(a)['content'], 'Edited')
        conflict = self.call('_save_preset', {'id': a, 'title': 'second', 'content': 'Must not save'})
        self.assertEqual(conflict.status, 409)
        self.assertEqual(self.db.get_card(a)['content'], 'Edited')
        self.assertEqual(self.db.get_card(b)['content'], 'Original')

    def test_existing_duplicates_can_be_edited_and_individually_deleted(self):
        a, b = self.seed('Existing'), self.seed('Existing')
        self.assertEqual(self.call('_save_preset', {'id': b, 'title': 'Existing', 'content': 'Edited duplicate'}).status, 200)
        deleted = self.call('_delete_preset', {'id': b})
        self.assertEqual(deleted.status, 200)
        self.assertIsNone(self.db.get_card(b))
        self.assertEqual(self.db.get_card(a)['content'], 'Original')

    def test_unrelated_library_cards_are_not_modified_or_deleted(self):
        other = self.seed('Other module', type='generation')
        self.assertEqual(self.call('_delete_preset', {'id': other}).status, 404)
        self.assertEqual(self.call('_save_preset', {'id': other, 'title': 'Changed', 'content': 'Text'}).status, 404)
        self.assertEqual(self.db.get_card(other)['content'], 'Original')
        self.assertEqual(self.call('_save_preset', {'title': 'Other module', 'content': 'A captioner prompt'}).status, 200)

    def test_delete_uses_library_attachment_cleanup(self):
        preset_id = self.seed('Attached', attachment='existing.png')
        self.assertEqual(self.call('_delete_preset', {'id': preset_id}).status, 200)
        self.assertEqual(self.cleaned, ['existing.png'])

    def test_unavailable_library_and_invalid_inputs_fail_without_writes(self):
        for data in [None, [], {'title': '', 'content': 'Text'}, {'title': 'Name', 'content': ''}, {'id': True, 'title': 'Name', 'content': 'Text'}]:
            self.assertEqual(self.call('_save_preset', data).status, 400)
        for value in [None, True, -1, 1.5, 'built:z_image']:
            self.assertEqual(self.call('_delete_preset', {'id': value}).status, 400)
        self.library = None
        self.assertEqual(self.call('_save_preset', {'title': 'Name', 'content': 'Text'}).status, 503)
        self.assertEqual(len(self.db.cards), 0)


if __name__ == '__main__':
    unittest.main()
