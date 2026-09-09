# CyberHub system-prompt JSON, schema 1

Captioner 1.6 adds a small format for sharing one system prompt, for
example as an attachment on Discord. It exports the current editor text rather
than the original preset. No captions, images, API addresses, model settings,
Library IDs or trigger-word field values are included. Placeholders in the
prompt itself, such as `{TRIGGER}` and `{WD_TAGS}`, are kept literally.

```json
{
  "schema": 1,
  "product": "CyberHub",
  "type": "system_prompt",
  "module": "captioner",
  "title": "Natural dataset caption",
  "prompt": "Describe only what is visible in the image.\nBegin with {TRIGGER}.\nReturn the caption only."
}
```

## Fields and compatibility

| Field | Required value or meaning |
| --- | --- |
| `schema` | Integer `1`, the format version, independent of the module version. |
| `product` | `CyberHub`. |
| `type` | `system_prompt`; distinguishes prompt files from installable packages and Library exports. |
| `module` | `captioner`; other module identifiers are rejected by Captioner. |
| `title` | Nonempty string, at most 200 characters after trimming surrounding whitespace. |
| `prompt` | Nonempty string, at most 100,000 UTF-16 code units. Content and line breaks are preserved. |

Files use UTF-8 JSON and a `.captioner.json` suffix. A UTF-8 BOM is accepted.
The whole file may be at most 512 KiB. Unknown fields are ignored and omitted
on re-export; required fields, module identity and schema version are validated.
Arrays and bare `{title, prompt}` objects are not accepted because they do not
identify a module or format version.

This envelope leaves room for Prompt Engineer to adopt the same product/type
markers with `module: "prompt_engineer"` and its own documented payload. That
module is not changed by this Captioner update. Captioner must never silently
treat a Prompt Engineer file as a Captioner prompt. A future incompatible
format should use a new schema version and an explicit migration.

## User workflow

- **Export JSON** downloads the title and current prompt text in the editor.
- **Import JSON** validates the file, then loads it as an imported draft. It
  replaces the previous imported draft, not a Library card. Invalid files leave
  the current editor unchanged.
- The imported draft remains in the dropdown when switching to other scripts.
  It and the active editor are restored in the same browser and origin, even
  with an empty image queue. Browser data clearing removes this local draft.
- **Save as preset** creates a new Captioner card in the optional Prompt Library
  using the editable title. Saving never overwrites an existing preset and
  rejects a name already used by a saved Captioner preset. Use Update to change
  the existing preset or enter a different title.
- **Update** explicitly updates the selected saved Library preset, including
  its title. Import/export themselves also work without Prompt Library.
- Import does not call a model, run the prompt, or change connection settings.

WD Hybrid prompts still require Auto Tagger to generate `{WD_TAGS}`; JSON shares
only the prompt text, not that dependency or a user's local generation settings.
