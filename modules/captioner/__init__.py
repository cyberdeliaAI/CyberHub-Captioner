"""VL Captioner — AI image captioning via local VLM (LM Studio, MLX, etc.)."""

import json
import os
import re
import threading
import unicodedata
from core import Module
from core.server import build_shell

DEFAULT_USER_INSTRUCTION = "Caption this image according to the system instructions."

DEFAULT_SYSTEM_PROMPT = """You are a captioning assistant for Z-Image LoRA training datasets.
Analyze the image and write one flowing natural-language caption.

Begin with the trigger token: {TRIGGER}

Describe only the variable elements: pose, expression, clothing, action, environment, background, framing, shot distance, angle, lighting direction and quality.

Do not describe traits that stay constant across the dataset. Those must be absorbed by the trigger token.

Caption only what is clearly visible. Write in plain declarative sentences. Keep it between 40 and 70 words.
No names, no celebrity identities, no invented context, no quality markers, no weighting syntax, no watermark or text disclaimers.
Output only the caption text."""


class CaptionerModule(Module):
    name = "Captioner"
    version = "1.6"
    release_stage = "stable"
    icon = "\U0001F4AC"   # 💬
    description = "Caption images using a local Vision Language Model"
    order = 35

    settings_schema = {}
    _preset_mutation_lock = threading.Lock()

    def routes_get(self):
        return {
            "/captioner": self._page,
            "/api/captioner/config": self._get_config,
            "/api/captioner/models": self._get_models,
            "/api/captioner/health": self._health,
            "/api/captioner/presets": self._get_presets,
        }

    def routes_post(self):
        return {
            "/api/captioner/config": self._save_config,
            "/api/captioner/caption": self._caption,
            "/api/captioner/sidecars": self._sidecars,
            "/api/captioner/presets/save": self._save_preset,
            "/api/captioner/presets/delete": self._delete_preset,
        }

    def _preset_library(self, handler):
        library = self.hub.registry.get("library")
        if library is None or getattr(library, "db", None) is None:
            handler.respond_json({"error": "Prompt Library is unavailable. Install or enable it to manage saved presets."}, status=503)
            return None
        return library

    @staticmethod
    def _preset_title_key(title):
        return " ".join(unicodedata.normalize("NFKC", str(title or "")).split()).casefold()

    @staticmethod
    def _is_captioner_preset(card):
        return bool(card and (card.get("type") == "captioner" or "captioner-preset" in (card.get("tags") or [])))

    @staticmethod
    def _all_library_presets(library):
        """Include legacy tagged presets and every page, not just the first 200."""
        cards = {}
        for filters in ({"type_filter": "captioner"}, {"tag_filter": "captioner-preset"}):
            offset = 0
            while True:
                result = library.db.list_cards(limit=200, offset=offset, **filters)
                batch = result.get("cards") or []
                cards.update((str(card["id"]), card) for card in batch)
                offset += len(batch)
                if not batch or offset >= result.get("total", offset):
                    break
        return sorted(cards.values(), key=lambda card: (card.get("updated_at") or 0, card["id"]), reverse=True)

    def _get_presets(self, handler, qs):
        library = self._preset_library(handler)
        if library is not None:
            handler.respond_json({"cards": self._all_library_presets(library)})

    @staticmethod
    def _preset_id(value):
        raw = str(value)
        if not re.fullmatch(r"[1-9][0-9]{0,18}", raw):
            return None
        number = int(raw)
        return number if number <= 9223372036854775807 else None

    def _save_preset(self, handler, content_len, content_type):
        data = handler.read_body_json(content_len)
        if not isinstance(data, dict):
            handler.respond_json({"error": "Invalid preset data"}, status=400)
            return
        title, content = data.get("title"), data.get("content")
        if (not isinstance(title, str) or not title.strip() or len(title.strip()) > 200
                or not isinstance(content, str) or not content.strip() or len(content) > 100000):
            handler.respond_json({"error": "Use a title of 1–200 characters and a nonempty prompt of at most 100,000 characters."}, status=400)
            return
        preset_id = self._preset_id(data["id"]) if "id" in data else None
        if "id" in data and preset_id is None:
            handler.respond_json({"error": "Invalid preset ID"}, status=400)
            return
        library = self._preset_library(handler)
        if library is None:
            return
        title = title.strip()
        # Serialize Captioner writes across tabs, including the name check. The
        # Library retains its own lock and data model; no other module is changed.
        with self._preset_mutation_lock:
            original = library.db.get_card(preset_id) if preset_id else None
            if preset_id and not self._is_captioner_preset(original):
                handler.respond_json({"error": "Captioner preset not found. Refresh the preset list."}, status=404)
                return
            key = self._preset_title_key(title)
            # Existing duplicate names may still be edited without renaming them.
            # Creating a preset or renaming one must not introduce a collision.
            if original is None or key != self._preset_title_key(original.get("title")):
                duplicate = next((card for card in self._all_library_presets(library)
                                  if card["id"] != preset_id and self._preset_title_key(card.get("title")) == key), None)
                if duplicate:
                    handler.respond_json({"error": "A saved Captioner preset with this name already exists. Choose another title or select that preset and use Update.",
                                          "duplicate_id": duplicate["id"]}, status=409)
                    return
            if original:
                library.db.update_card(preset_id, {"title": title, "content": content,
                                                  "type": "captioner", "target": original.get("target") or "general"})
            else:
                preset_id = library.db.create_card({"title": title, "content": content,
                                                   "type": "captioner", "target": "general", "tags": ["captioner-preset"]})
            handler.respond_json({"ok": True, "id": preset_id, "preset": library.db.get_card(preset_id)})

    def _delete_preset(self, handler, content_len, content_type):
        data = handler.read_body_json(content_len)
        preset_id = self._preset_id(data.get("id")) if isinstance(data, dict) else None
        if preset_id is None:
            handler.respond_json({"error": "Invalid preset ID"}, status=400)
            return
        library = self._preset_library(handler)
        if library is None:
            return
        with self._preset_mutation_lock:
            if not self._is_captioner_preset(library.db.get_card(preset_id)):
                handler.respond_json({"error": "Captioner preset not found. Refresh the preset list."}, status=404)
                return
            old_attachment = library.db.delete_card(preset_id)
        if old_attachment:
            library._cleanup_attachment_if_unused(old_attachment)
        handler.respond_json({"ok": True, "id": preset_id})

    def _get_cfg(self):
        temperature = self._optional_number(self.setting("temperature"))
        top_p = self._optional_number(self.setting("top_p"))
        max_tokens = self._optional_number(self.setting("max_tokens"), integer=True)
        top_k = self._optional_number(self.setting("top_k"), integer=True)
        presence_penalty = self._optional_number(self.setting("presence_penalty"))
        return {
            "api_url": self._normalize_api_url(self.setting("api_url")),
            "model": self.setting("model") or "",
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
        }

    @staticmethod
    def _normalize_api_url(value):
        url = str(value or "http://localhost:1234").strip()
        if not re.match(r"^https?://", url, re.IGNORECASE):
            url = "http://" + url
        url = url.rstrip("/")
        if not re.search(r"/v1$", url, re.IGNORECASE):
            url += "/v1"
        return url

    @staticmethod
    def _optional_number(value, integer=False):
        if value is None or str(value).strip() == "":
            return None
        try:
            return int(value) if integer else float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _render_trigger_prompt(prompt, trigger_word):
        trigger = " ".join(str(trigger_word or "").split())[:200]
        if trigger:
            if re.search(r"\{TRIGGER\}", prompt, re.IGNORECASE):
                return re.sub(r"\{TRIGGER\}", lambda _match: trigger, prompt,
                              flags=re.IGNORECASE), trigger
            instruction = f"Begin the final caption with this exact trigger token: {trigger}"
            return f"{prompt.rstrip()}\n\n{instruction}", trigger

        lines = [line for line in prompt.splitlines()
                 if not re.search(r"\{TRIGGER\}", line, re.IGNORECASE)]
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip(), ""

    @staticmethod
    def _ensure_trigger_prefix(caption, trigger):
        text = str(caption or "").strip()
        prefix_match = trigger and re.match(
            r"^" + re.escape(trigger) + r"(?=$|[\s,.:;!?])", text
        )
        if not trigger:
            return text
        if prefix_match:
            remainder = re.sub(r"^[\s,.:;!?-]+", "", text[prefix_match.end():])
            return f"{trigger}, {remainder}" if remainder else trigger
        return f"{trigger}, {text}" if text else trigger

    def _page(self, handler, qs):
        html = build_shell(self.hub.registry, self.hub.settings,
            active_key="captioner", page_title="Captioner", body_html=PAGE_BODY)
        handler.respond_html(html)

    def _get_config(self, handler, qs):
        handler.respond_json(self._get_cfg())

    def _save_config(self, handler, content_len, content_type):
        data = handler.read_body_json(content_len)
        if data is None:
            handler.respond_json({"error": "Invalid JSON"}, status=400); return
        for key in ("api_url", "model", "temperature", "top_p", "max_tokens",
                    "top_k", "presence_penalty"):
            if key in data:
                value = data[key]
                if key == "api_url":
                    value = self._normalize_api_url(value)
                elif key in ("temperature", "top_p", "max_tokens", "top_k", "presence_penalty"):
                    value = "" if value is None or str(value).strip() == "" else value
                self.hub.settings.set_module_setting("captioner", key, value)
        handler.respond_json({"ok": True})

    def _get_models(self, handler, qs):
        import requests
        cfg = self._get_cfg()
        try:
            resp = requests.get(f"{cfg['api_url']}/models", timeout=5)
            resp.raise_for_status()
            models = [m["id"] for m in resp.json().get("data", [])]
            handler.respond_json({"models": models})
        except Exception as e:
            handler.respond_json({"error": str(e), "models": []}, status=503)

    def _health(self, handler, qs):
        import requests
        cfg = self._get_cfg()
        try:
            resp = requests.get(f"{cfg['api_url']}/models", timeout=3)
            resp.raise_for_status()
            models = resp.json().get("data", [])
            if not models: handler.respond_json({"ok": False, "reason": "no_models"}, status=503); return
            handler.respond_json({"ok": True, "count": len(models)})
        except Exception:
            handler.respond_json({"ok": False, "reason": "no_connection"}, status=503)

    def _caption(self, handler, content_len, content_type):
        import requests
        data = handler.read_body_json(content_len)
        if data is None: handler.respond_json({"error": "Invalid JSON"}, status=400); return
        image_b64 = data.get("image_b64")
        if not image_b64: handler.respond_json({"error": "No image provided"}, status=400); return
        cfg = self._get_cfg()
        media_type = data.get("media_type", "image/jpeg")
        override = (data.get("override_prompt") or "").strip()
        system_prompt, trigger = self._render_trigger_prompt(
            override if override else cfg["system_prompt"], data.get("trigger_word")
        )
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_b64}"}},
                    {"type": "text", "text": DEFAULT_USER_INSTRUCTION},
                ]},
            ],
        }
        if cfg["model"]:
            payload["model"] = cfg["model"]
        for key in ("temperature", "top_p", "max_tokens", "top_k", "presence_penalty"):
            if cfg[key] is not None:
                payload[key] = cfg[key]
        try:
            resp = requests.post(f"{cfg['api_url']}/chat/completions", json=payload, timeout=120)
            if (not resp.ok and re.search(
                    r"top_k|presence_penalty|unsupported|unknown.*(param|field|key)",
                    resp.text or "", re.IGNORECASE)):
                fallback_payload = dict(payload)
                fallback_payload.pop("top_k", None)
                fallback_payload.pop("presence_penalty", None)
                resp = requests.post(f"{cfg['api_url']}/chat/completions",
                                     json=fallback_payload, timeout=120)
            resp.raise_for_status()
            caption_text = self._ensure_trigger_prefix(
                resp.json()["choices"][0]["message"]["content"], trigger
            )
            handler.respond_json({"caption": caption_text})
        except requests.exceptions.ConnectionError:
            handler.respond_json({
                "error": f"Cannot connect to {cfg['api_url']}. Is the LM Studio server running?"
            }, status=503)
        except requests.exceptions.Timeout:
            handler.respond_json({"error": "Request timed out after 120s."}, status=504)
        except (KeyError, IndexError, ValueError) as e:
            handler.respond_json({"error": f"Unexpected API response: {e}"}, status=502)
        except requests.exceptions.RequestException as e:
            handler.respond_json({"error": str(e)}, status=500)

    def _sidecars(self, handler, content_len, content_type):
        """Build separate training-caption sidecars for browser-only clients."""
        import io
        import zipfile

        data = handler.read_body_json(content_len)
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list) or not items:
            handler.respond_json({"error": "No captions provided"}, status=400)
            return
        if len(items) > 10000:
            handler.respond_json({"error": "Too many captions (maximum 10000)"}, status=400)
            return

        total_chars = 0
        used = set()
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for index, item in enumerate(items, start=1):
                if not isinstance(item, dict):
                    continue
                source_name = str(item.get("name") or "").replace("\x00", "").replace("\\", "/").rsplit("/", 1)[-1]
                stem = os.path.splitext(source_name)[0].strip() or f"caption_{index}"
                caption = str(item.get("caption") or "").strip()
                if not caption:
                    continue
                total_chars += len(caption)
                if total_chars > 10_000_000:
                    handler.respond_json({"error": "Caption export is too large"}, status=413)
                    return

                filename = f"{stem}.txt"
                suffix = 2
                while filename.casefold() in used:
                    filename = f"{stem}_{suffix}.txt"
                    suffix += 1
                used.add(filename.casefold())
                archive.writestr(filename, caption + "\n")

        if not used:
            handler.respond_json({"error": "No completed captions provided"}, status=400)
            return
        handler.respond_binary(output.getvalue(), "application/zip",
                               download_name="caption-files.zip")

PAGE_BODY = r"""
<style>
/* VL Captioner — module-scoped styles. All rules live under .cap-app so the
   hub's :root variables stay intact and class names don't leak into the topbar
   or other modules. */

.cap-app, .cap-app *, .cap-app *::before, .cap-app *::after { box-sizing: border-box; }

.cap-app {
    height: calc(100vh - 48px);
    display: flex;
    flex-direction: column;
    background: var(--bg-darkest);
    color: var(--text);
    font-family: var(--font);
    font-size: 13px;
    min-height: 0;
    overflow: hidden;
}

.cap-status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--text-dim); transition: background 0.3s;
}
.cap-status-dot.connected { background: var(--green); box-shadow: 0 0 6px var(--green); }
.cap-status-dot.error { background: var(--red); }

.cap-status-label {
    font-family: var(--mono); font-size: 11px;
    color: var(--text-dim);
}
.cap-app .cap-btn {
    font: inherit; font-size: 12px; font-weight: 400;
    padding: 7px 12px;
    border: 1px solid var(--border-light);
    background: var(--bg-card);
    color: var(--text);
    cursor: pointer; border-radius: 6px;
    transition: border-color .15s, color .15s, background .15s;
    white-space: nowrap;
}
.cap-app .cap-btn:hover:not(:disabled) { border-color: var(--accent); color: var(--text-bright); }
.cap-app .cap-btn:active:not(:disabled) { transform: scale(0.97); }
.cap-app .cap-btn.primary {
    background: var(--accent); border-color: var(--accent); color: #fff;
}
.cap-app .cap-btn.primary:hover:not(:disabled) { background: var(--accent-dim); border-color: var(--accent-dim); color: #fff; }
.cap-app .cap-btn.danger { border-color: rgba(239,68,68,.4); color: var(--red); }
.cap-app .cap-btn.danger:hover:not(:disabled) { border-color: var(--red); background: rgba(239,68,68,.08); }
.cap-app .cap-btn:disabled { opacity: 0.4; cursor: not-allowed; }

.cap-main { display: flex; flex: 1; overflow: hidden; min-height: 0; }

.cap-sidebar {
    width: 284px; min-width: 284px;
    border-right: 1px solid var(--border);
    background: var(--bg-panel);
    display: flex; flex-direction: column;
    overflow: hidden;
    min-height: 0;
}
.cap-sidebar-scroll {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    overflow-x: hidden;
    scrollbar-width: thin;
    scrollbar-color: var(--border-light) transparent;
}
.cap-sidebar-scroll::-webkit-scrollbar,
.cap-queue::-webkit-scrollbar { width: 4px; }
.cap-sidebar-scroll::-webkit-scrollbar-track,
.cap-queue::-webkit-scrollbar-track { background: transparent; }
.cap-sidebar-scroll::-webkit-scrollbar-thumb,
.cap-queue::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 2px; }
.cap-sidebar-section {
    border-bottom: 1px solid var(--border);
    padding: 16px;
}
.cap-sidebar-label {
    font-family: var(--mono);
    font-size: 10px; letter-spacing: 0.12em;
    color: var(--text-dim);
    text-transform: uppercase;
    margin-bottom: 10px;
    font-weight: 700;
}
.cap-field {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-top: 10px;
}
.cap-field:first-of-type { margin-top: 0; }
.cap-field label {
    font-size: 11px;
    color: var(--text-dim);
}
.cap-field-note {
    color: var(--text-dim);
    font-size: 10px;
    line-height: 1.45;
}
.cap-app .cap-input {
    width: 100%;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-family: var(--font);
    font-size: 12px;
    padding: 7px 9px;
    outline: none;
    transition: border-color 0.15s;
}
.cap-app .cap-input:focus { border-color: var(--accent); }
.cap-app .cap-input::placeholder { color: var(--text-dim); }
.cap-drop {
    border: 1px dashed var(--border-light);
    border-radius: 6px;
    padding: 20px 16px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    color: var(--text-dim);
    font-size: 12px; line-height: 1.6;
    background: var(--bg-card);
}
.cap-drop:hover, .cap-drop.drag-over {
    border-color: var(--accent);
    color: var(--text);
    background: var(--accent-glow);
}
.cap-drop svg { display: block; margin: 0 auto 8px; opacity: 0.5; }
.cap-input-actions {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
}
.cap-input-actions .cap-btn { flex: 0 0 auto; }
.cap-sidecar-toggle {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    min-width: 0;
    color: var(--text-dim);
    font-size: 10px;
    cursor: pointer;
}
.cap-sidecar-toggle input { accent-color: var(--accent); }
.cap-sidecar-status {
    margin-top: 7px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.cap-queue {
    flex: 1 1 auto;
    overflow-y: auto;
    overflow-x: hidden;
    min-height: 0;
    padding: 8px;
    scrollbar-width: thin;
    scrollbar-color: var(--border-light) transparent;
}
.cap-queue:empty {
    margin-top: 0;
}

.cap-queue-item {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 8px;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.1s;
    border: 1px solid transparent;
    margin-bottom: 2px;
}
.cap-queue-item:hover { background: var(--bg-hover); }
.cap-queue-item.active { background: var(--bg-active); border-color: var(--accent-dim); }

.cap-queue-thumb {
    width: 36px; height: 36px;
    border-radius: 3px;
    object-fit: cover;
    background: var(--bg-card);
    flex-shrink: 0;
}
.cap-queue-info { flex: 1; min-width: 0; }
.cap-queue-name {
    font-size: 12px; color: var(--text);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.cap-queue-status {
    font-family: var(--mono); font-size: 10px;
    color: var(--text-dim); margin-top: 2px;
}
.cap-queue-status.done { color: var(--green); }
.cap-queue-status.error { color: var(--red); }
.cap-queue-status.running { color: var(--accent); }

.cap-sidebar-footer {
    padding: 12px 16px;
    border-top: 1px solid var(--border);
    display: flex; flex-direction: column; gap: 8px;
    flex: 0 0 auto;
    background: var(--bg-panel);
}
.cap-progress-bar {
    height: 3px; background: var(--bg-card);
    border-radius: 2px; overflow: hidden;
}
.cap-progress-fill {
    height: 100%; background: var(--accent);
    border-radius: 2px;
    transition: width 0.3s; width: 0%;
}
.cap-progress-text {
    font-family: var(--mono); font-size: 10px;
    color: var(--text-dim);
}
.cap-footer-summary {
    min-width: 0;
}
.cap-batch-actions {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 6px;
    width: 100%;
}
.cap-batch-actions .cap-btn {
    width: 100%;
    min-width: 0;
    padding-left: 8px;
    padding-right: 8px;
}
.cap-batch-actions #runUncaptionedBtn {
    grid-column: 1 / -1;
}

.cap-content { flex: 1; display: flex; flex-direction: column; overflow: hidden; min-width: 0; }
.cap-viewer { flex: 1; display: flex; overflow: hidden; min-height: 0; }

.cap-image-panel {
    flex: 1; display: flex; align-items: center; justify-content: center;
    background: var(--bg-darkest);
    padding: 32px;
    position: relative; overflow: hidden;
    min-width: 0;
}
.cap-image-panel img {
    max-width: 100%; max-height: 100%;
    object-fit: contain;
    border-radius: 4px;
    display: block;
    box-shadow: 0 8px 32px rgba(0,0,0,.18);
}
.cap-empty {
    text-align: center;
    color: var(--text-dim);
}
.cap-empty svg { margin: 0 auto 16px; opacity: 0.3; display: block; }
.cap-empty p { font-size: 13px; line-height: 1.7; }

.cap-caption-panel {
    width: 36%; min-width: 320px;
    border-left: 1px solid var(--border);
    background: var(--bg-panel);
    display: flex; flex-direction: column;
    overflow: hidden;
}
.cap-panel-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
}
.cap-panel-title {
    font-family: var(--mono);
    font-size: 10px; letter-spacing: 0.12em;
    color: var(--text-dim); text-transform: uppercase;
    font-weight: 700;
}
.cap-panel-actions { display: flex; gap: 6px; }

.cap-caption-area {
    flex: 1; padding: 16px; overflow-y: auto; min-height: 0;
}
.cap-app .cap-textarea {
    width: 100%; height: 100%;
    min-height: 120px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-family: var(--font);
    font-size: 14px; line-height: 1.8;
    padding: 12px;
    resize: none; outline: none;
    transition: border-color 0.15s;
}
.cap-app .cap-textarea:focus { border-color: var(--accent); }
.cap-app .cap-textarea::placeholder { color: var(--text-dim); }

/* Preset selector in the sidebar */
.cap-app .cap-select {
    width: 100%;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-size: 12px;
    padding: 6px 8px;
    outline: none;
    transition: border-color 0.15s;
}
.cap-app .cap-select:focus { border-color: var(--accent); }
.cap-app .cap-preset-row {
    display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px;
}
.cap-app .cap-preset-row .cap-btn { flex: 1; font-size: 11px; padding: 5px 8px; }
.cap-app .cap-preset-editor {
    height: 170px;
    min-height: 120px;
    max-height: 220px;
    margin-top: 8px;
    font-size: 11px; line-height: 1.5;
    display: none;
}
.cap-app .cap-preset-editor.active { display: block; }
.cap-hybrid-controls {
    display: none;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid var(--border);
}
.cap-hybrid-controls.active { display: block; }
.cap-hybrid-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 72px;
    gap: 8px;
    align-items: end;
}
.cap-hybrid-row .cap-field { min-width: 0; }

/* Target dropdown sits inline with the save buttons */
.cap-app .cap-target-label {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11px; color: var(--text-dim);
    margin: 0 4px 0 8px;
}
.cap-app .cap-target-select {
    width: auto; padding: 4px 6px; font-size: 11px;
}

.cap-caption-meta {
    padding: 10px 16px;
    border-top: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 10px; color: var(--text-dim);
    display: flex; justify-content: space-between; align-items: center;
}

.cap-toolbar {
    padding: 10px 16px;
    border-top: 1px solid var(--border);
    background: var(--bg-panel);
    display: flex; gap: 8px; align-items: center; justify-content: space-between;
    flex-shrink: 0;
}
.cap-toolbar-left, .cap-toolbar-right { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }

.cap-spinner {
    display: inline-block;
    width: 12px; height: 12px;
    border: 1.5px solid var(--border-light);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: cap-spin 0.7s linear infinite;
    vertical-align: middle;
    margin-right: 4px;
}
@keyframes cap-spin { to { transform: rotate(360deg); } }

.cap-toast {
    position: fixed;
    bottom: 24px; right: 22px;
    background: var(--bg-panel);
    border: 1px solid var(--border-light);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 10px 16px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text);
    z-index: 9999;
    opacity: 0;
    transform: translateY(8px);
    transition: all 0.2s;
    pointer-events: none;
    box-shadow: 0 8px 24px rgba(0,0,0,.45);
}
.cap-toast.show { opacity: 1; transform: translateY(0); }
.cap-toast.success { border-left-color: var(--green); }
.cap-toast.error-toast { border-left-color: var(--red); }

.cap-app input[type="file"] { display: none; }

.cap-workspace-header { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:16px 22px; border-bottom:1px solid var(--border); background:var(--bg-panel); flex-shrink:0; }
.cap-workspace-header h1 { font-size:18px; font-weight:600; letter-spacing:-.3px; color:var(--text-bright); line-height:1.3; }
.cap-workspace-header p { margin-top:3px; font-size:12px; color:var(--text-dim); }
.cap-settings-link { display:inline-flex; align-items:center; gap:8px; background:none; border:0; color:var(--accent); font:inherit; cursor:pointer; padding:8px 0 8px 12px; }
.cap-settings-link:hover { color:var(--text-bright); text-decoration:underline; }
.cap-app :is(button, select, input, textarea):focus-visible { outline:2px solid var(--accent); outline-offset:3px; }
.cap-connection-section { padding:12px; }
.cap-connection-trigger { width:100%; display:flex; align-items:center; gap:10px; border:1px solid transparent; border-radius:8px; padding:8px; background:transparent; color:var(--text); font:inherit; text-align:left; cursor:pointer; }
.cap-connection-trigger:hover { border-color:var(--border-light); background:var(--bg-hover); }
.cap-connection-copy { display:flex; flex-direction:column; gap:4px; flex:1; min-width:0; }
.cap-connection-title { color:var(--text-bright); font-size:12px; font-weight:500; }
.cap-status-dot { flex-shrink:0; }
.cap-status-label { font-family:var(--font); font-size:11px; }
.cap-connection-arrow { color:var(--text-dim); font-size:22px; }
.cap-sidebar-label { display:block; letter-spacing:.08em; }
.cap-drop { display:block; width:100%; font-family:inherit; }
.cap-input-actions { flex-wrap:wrap; }
.cap-sidecar-toggle { font-size:11px; }
.cap-sidecar-status { white-space:normal; }
.cap-preset-editor { scrollbar-width:thin; }
.cap-prompt-transfer { margin-top:8px; }
.cap-prompt-transfer[data-error="true"] { color:var(--red); }
.cap-queue-column { width:196px; min-width:196px; display:flex; flex-direction:column; min-height:0; border-right:1px solid var(--border); background:var(--bg-panel); }
.cap-queue-header { display:flex; align-items:center; justify-content:space-between; padding:16px 14px; border-bottom:1px solid var(--border); }
.cap-queue-count { font:11px var(--mono); color:var(--text-dim); background:var(--bg-card); border:1px solid var(--border); border-radius:5px; padding:1px 6px; }
.cap-queue-empty { padding:18px 6px; color:var(--text-dim); font-size:11px; line-height:1.7; }
.cap-queue-item { width:100%; font:inherit; text-align:left; color:var(--text); background:transparent; border-radius:6px; padding:8px 6px; }
.cap-queue-thumb { width:38px; height:44px; border-radius:4px; }
.cap-queue-name { font-size:11px; }
.cap-queue-status { font-family:var(--font); font-size:10px; }
.cap-queue-status.pending { color:var(--text-dim); }
.cap-queue-status.edited { color:var(--orange); }
.cap-sidebar-footer { padding:12px; }
.cap-progress-bar { height:4px; }
.cap-image-panel { padding:24px; }
.cap-image-panel img { min-height:0; }
.cap-panel-header { gap:10px; flex-wrap:wrap; }
.cap-panel-title { letter-spacing:.08em; }
.cap-caption-area { padding:16px; display:flex; overflow:hidden; }
.cap-caption-area .cap-textarea { flex:1; min-height:0; }
.cap-edit-feedback { padding:0 16px 14px; font-size:11px; color:var(--text-dim); line-height:1.6; }
.cap-edit-feedback #captionSaveStatus { color:var(--text); display:block; margin-bottom:4px; }
.cap-edit-feedback #captionSaveStatus[data-error="true"] { color:var(--red); }
.cap-caption-meta { gap:12px; }
#currentFile { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; text-align:right; }
#tokenCount { flex-shrink:0; }
.cap-toolbar { flex-wrap:wrap; gap:12px 20px; padding:12px 16px; }
.cap-toolbar-label { color:var(--text-dim); font-size:11px; }
.cap-app .cap-target-label { margin:0; }
.cap-settings-dialog { margin:auto; width:min(600px, calc(100vw - 32px)); max-height:calc(100dvh - 40px); padding:0; background:var(--bg-panel); color:var(--text); border:1px solid var(--border-light); border-radius:14px; box-shadow:0 24px 80px rgba(0,0,0,.5); font:13px var(--font); }
.cap-settings-dialog::backdrop { background:rgba(0,0,0,.65); backdrop-filter:blur(4px); }
.cap-settings-dialog form { display:flex; flex-direction:column; max-height:calc(100dvh - 42px); }
.cap-dialog-header { padding:22px 24px 18px; border-bottom:1px solid var(--border); display:flex; justify-content:space-between; align-items:flex-start; gap:16px; }
.cap-dialog-header h2 { font-size:18px; font-weight:600; color:var(--text-bright); }
.cap-dialog-header p, .cap-settings-section p { font-size:12px; line-height:1.6; color:var(--text-dim); margin-top:4px; }
.cap-app .cap-close { font-size:22px; padding:0 8px; background:transparent; border-color:transparent; }
.cap-dialog-body { overflow-y:auto; padding:20px 24px; }
.cap-settings-section { margin-bottom:14px; }
.cap-settings-section h3 { font-size:12px; font-weight:600; color:var(--text-bright); }
.cap-dialog-body > .cap-field + .cap-settings-section { margin-top:24px; padding-top:20px; border-top:1px solid var(--border); }
.cap-settings-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px 16px; margin-bottom:16px; }
.cap-settings-grid .cap-field { margin:0; }
.cap-settings-dialog .cap-input { padding:9px 11px; }
.cap-settings-dialog .cap-field label { color:var(--text); font-size:12px; }
.cap-settings-dialog .cap-field-note { font-size:11px; }
.cap-settings-feedback { margin-top:12px; font-size:12px; line-height:1.5; color:var(--text); }
.cap-settings-feedback:empty { display:none; }
.cap-settings-feedback[data-error="true"] { color:var(--red); }
.cap-dialog-footer { padding:16px 24px; border-top:1px solid var(--border); display:flex; justify-content:space-between; flex-wrap:wrap; gap:12px; }
.cap-dialog-footer > div { display:flex; gap:8px; }
@media (max-width: 1250px) {
    .cap-sidebar { width:260px; min-width:260px; }
    .cap-queue-column { width:180px; min-width:180px; }
    .cap-caption-panel { width:42%; min-width:300px; }
    .cap-image-panel { padding:18px; }
}
@media (max-width: 1080px) {
    .cap-viewer { flex-direction:column; }
    .cap-image-panel { flex:1; min-height:160px; }
    .cap-caption-panel { flex:1; width:auto; min-width:0; min-height:240px; border-left:0; border-top:1px solid var(--border); }
}
@media (max-width: 720px) {
    .cap-app { height:auto; min-height:calc(100dvh - 48px); overflow:visible; }
    .cap-workspace-header { padding:14px 16px; }
    .cap-workspace-header p { max-width:220px; }
    .cap-main { flex-direction:column; overflow:visible; }
    .cap-sidebar { width:100%; min-width:0; border-right:0; }
    .cap-sidebar-scroll { overflow:visible; }
    .cap-queue-column { width:100%; min-width:0; height:300px; border-right:0; border-bottom:1px solid var(--border); }
    .cap-content { overflow:visible; }
    .cap-viewer { flex:none; height:700px; }
    .cap-settings-grid { grid-template-columns:1fr 1fr; }
    .cap-dialog-header, .cap-dialog-body, .cap-dialog-footer { padding:16px; }
}
</style>
<div class="cap-app">
<header class="cap-workspace-header">
    <div><h1>Captioner</h1><p>Prepare, caption and refine your image dataset.</p></div>
    <button class="cap-settings-link" id="connectionSettingsLink" type="button" onclick="openConnectionSettings()" aria-haspopup="dialog" aria-controls="connectionDialog">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M4 7h16M4 17h16"/><circle cx="9" cy="7" r="3" fill="var(--bg-panel)"/><circle cx="15" cy="17" r="3" fill="var(--bg-panel)"/></svg>
        Settings
    </button>
</header>

<div class="cap-main">
    <div class="cap-sidebar">
        <div class="cap-sidebar-scroll">
        <div class="cap-sidebar-section cap-connection-section">
            <button class="cap-connection-trigger" id="connectionTrigger" type="button" onclick="openConnectionSettings()" aria-haspopup="dialog" aria-controls="connectionDialog" title="Open Captioner settings">
                <span class="cap-status-dot" id="statusDot"></span>
                <span class="cap-connection-copy"><span class="cap-connection-title">Connection</span><span class="cap-status-label" id="statusLabel" aria-live="polite">Checking connection…</span></span>
                <span class="cap-connection-arrow" aria-hidden="true">›</span>
            </button>
        </div>

        <div class="cap-sidebar-section">
            <div class="cap-sidebar-label">Input</div>
            <button type="button" class="cap-drop" id="dropZone" onclick="document.getElementById('fileInput').click()">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <rect x="3" y="3" width="18" height="18" rx="2"/>
                    <circle cx="8.5" cy="8.5" r="1.5"/>
                    <polyline points="21 15 16 10 5 21"/>
                </svg>
                Drop images here<br>or click to browse
            </button>
            <input type="file" id="fileInput" accept="image/*" multiple>
            <input type="file" id="folderInput" accept="image/*" multiple webkitdirectory>
            <div class="cap-input-actions">
                <button class="cap-btn" id="openFolderBtn" onclick="openImageFolder()">Open folder</button>
                <label class="cap-sidecar-toggle" title="Write one matching .txt training caption for every completed image">
                    <input id="saveSidecarsToggle" type="checkbox" checked>
                    Save .txt beside images
                </label>
            </div>
            <div class="cap-field-note cap-sidecar-status" id="sidecarFolderStatus"></div>
        </div>

        <div class="cap-sidebar-section">
            <label class="cap-sidebar-label" for="presetSelect">Vision script</label>
            <select id="presetSelect" class="cap-select" onchange="onPresetChange()" title="Pick the vision instruction sent to the model before each caption request.">
                <option value="built:z_image">Z-Image LoRA training</option>
            </select>
            <div class="cap-field">
                <label for="presetTitle">Prompt title</label>
                <input class="cap-input" id="presetTitle" type="text" maxlength="200" placeholder="Name your system prompt">
            </div>
            <textarea id="presetEditor" class="cap-textarea cap-preset-editor" rows="4" aria-label="System prompt" placeholder="Vision script content..." title="Edit the active script. Save as preset to keep your changes."></textarea>
            <div class="cap-preset-row">
                <button class="cap-btn" id="savePresetBtn" onclick="saveAsPreset()" title="Save the current script as a new Library preset">Save as preset</button>
                <button class="cap-btn" id="updatePresetBtn" onclick="updatePreset()" style="display:none" title="Overwrite the selected preset">Update</button>
                <button class="cap-btn danger" id="deletePresetBtn" onclick="deletePreset()" style="display:none" title="Delete the selected saved preset from the Library">Delete preset</button>
            </div>
            <div class="cap-preset-row">
                <button class="cap-btn" id="importPromptBtn" type="button" onclick="document.getElementById('promptJsonInput').click()">Import JSON</button>
                <button class="cap-btn" id="exportPromptBtn" type="button" onclick="exportPromptJson()" title="Download the current title and system prompt">Export JSON</button>
            </div>
            <input type="file" id="promptJsonInput" accept=".json,application/json" onchange="importPromptFile(this)">
            <div class="cap-field-note cap-prompt-transfer" id="promptTransferStatus" role="status">Share a title and system prompt as JSON.</div>
            <div class="cap-field">
                <label for="capTriggerWord">Trigger word (optional)</label>
                <input class="cap-input" id="capTriggerWord" type="text" maxlength="200" spellcheck="false" placeholder="e.g. ohwx_person">
                <div class="cap-field-note">Replaces <code>{TRIGGER}</code> in the selected script. Leave empty to remove that instruction.</div>
            </div>
            <div class="cap-hybrid-controls" id="wdHybridControls">
                <div class="cap-hybrid-row">
                    <div class="cap-field">
                        <label for="capConstantTraits">Constant WD tags</label>
                        <input class="cap-input" id="capConstantTraits" type="text" spellcheck="false" placeholder="blonde_hair, blue_eyes">
                    </div>
                    <div class="cap-field">
                        <label for="capWdThreshold">WD score</label>
                        <input class="cap-input" id="capWdThreshold" type="number" min="0.05" max="0.95" step="0.05" value="0.40">
                    </div>
                </div>
                <div class="cap-field-note" id="wdHybridStatus">Checking Auto Tagger...</div>
            </div>
        </div>
        </div>

    </div>

    <aside class="cap-queue-column" aria-label="Image queue">
        <div class="cap-queue-header"><h2 class="cap-panel-title">Queue</h2><span class="cap-queue-count" id="queueCount">0</span></div>
        <div class="cap-queue" id="queueList" aria-label="Queued images"><div class="cap-queue-empty">Your images will appear here.<br>Add files or open a folder to start.</div></div>
        <div class="cap-sidebar-footer">
            <div class="cap-progress-bar"><div class="cap-progress-fill" id="progressFill"></div></div>
            <div class="cap-footer-summary">
                <span class="cap-progress-text" id="progressText">no images loaded</span>
            </div>
            <div class="cap-batch-actions">
                    <button class="cap-btn primary" id="runUncaptionedBtn" disabled onclick="runUncaptioned()" title="Caption only images that do not have a caption yet">Run uncaptioned</button>
                    <button class="cap-btn" id="runAllBtn" disabled onclick="runAll()">Run all</button>
                    <button class="cap-btn danger" id="stopAllBtn" style="display:none" onclick="stopAll()">Stop</button>
                    <button class="cap-btn danger" id="clearBtn" disabled onclick="clearAll()">Clear</button>
            </div>
        </div>
    </aside>

    <div class="cap-content">
        <div class="cap-viewer">
            <div class="cap-image-panel" id="imagePanel">
                <div class="cap-empty">
                    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
                        <rect x="3" y="3" width="18" height="18" rx="2"/>
                        <circle cx="8.5" cy="8.5" r="1.5"/>
                        <polyline points="21 15 16 10 5 21"/>
                    </svg>
                    <p>No image selected.<br>Add images from the sidebar.</p>
                </div>
            </div>

            <div class="cap-caption-panel">
                <div class="cap-panel-header">
                    <span class="cap-panel-title">Caption output</span>
                    <div class="cap-panel-actions">
                        <button class="cap-btn" id="copyOneBtn" onclick="copyCaption()" title="Copy current caption">Copy</button>
                        <button class="cap-btn primary" id="runOneBtn" disabled onclick="runOne()">Caption this</button>
                    </div>
                </div>
                <div class="cap-caption-area">
                    <textarea class="cap-textarea" id="captionOutput" aria-label="Caption output" aria-describedby="captionEditNote captionSaveStatus" disabled placeholder="Caption will appear here..." spellcheck="false"></textarea>
                </div>
                <div class="cap-edit-feedback">
                    <span id="captionSaveStatus" role="status">Select an image to edit its caption.</span>
                    <p id="captionEditNote">Edits are kept in this browser. Save caption files writes the latest text.</p>
                </div>
                <div class="cap-caption-meta">
                    <span id="tokenCount">— words</span>
                    <span id="currentFile">no file selected</span>
                </div>
            </div>
        </div>

        <div class="cap-toolbar">
            <div class="cap-toolbar-left">
                <span class="cap-toolbar-label">Library</span>
                <label class="cap-target-label" title="Tags saved cards with this target so Library filters work">Target
                    <select id="saveTargetSelect" class="cap-select cap-target-select">
                        <option value="general">General</option>
                        <option value="sdxl">SDXL</option>
                        <option value="zit">Z-Image</option>
                        <option value="flux">Flux</option>
                        <option value="illustrious">Illustrious</option>
                        <option value="pony">Pony</option>
                        <option value="llm">LLM</option>
                        <option value="suno">Suno</option>
                    </select>
                </label>
                <button class="cap-btn" onclick="saveCurrentToLibrary()">Save current</button>
                <button class="cap-btn" onclick="saveAllToLibrary()">Save all</button>
            </div>
            <div class="cap-toolbar-right">
                <button class="cap-btn" onclick="copyAll()">Copy all</button>
                <button class="cap-btn primary" id="saveCaptionFilesBtn" disabled onclick="saveSidecarsNow()">Save caption files</button>
                <button class="cap-btn" onclick="exportTxt()">Export .txt</button>
                <button class="cap-btn" onclick="exportCsv()">Export .csv</button>
            </div>
        </div>
    </div>
</div>

<dialog class="cap-settings-dialog" id="connectionDialog" aria-labelledby="connectionDialogTitle" aria-describedby="connectionDialogDescription">
    <form id="connectionForm" onsubmit="event.preventDefault(); saveConnectionConfig()">
        <header class="cap-dialog-header"><div><h2 id="connectionDialogTitle">Captioner settings</h2><p id="connectionDialogDescription">Connect your vision model and adjust generation.</p></div><button class="cap-btn cap-close" type="button" onclick="closeConnectionSettings()" aria-label="Close settings">×</button></header>
        <div class="cap-dialog-body">
            <div class="cap-settings-section"><h3>Connection</h3><p>Use a server with a vision-capable model loaded.</p></div>
                <div class="cap-field">
                    <label for="capApiUrl">API URL</label>
                    <input class="cap-input" id="capApiUrl" type="text" required spellcheck="false" placeholder="http://localhost:1234">
                    <div class="cap-field-note">Your LM Studio or OpenAI-compatible server. With or without <code>/v1</code>.</div>
                </div>
                <div class="cap-field">
                    <label for="capModel">Model</label>
                    <input class="cap-input" id="capModel" type="text" spellcheck="false" placeholder="LM Studio default">
                </div>
                <div class="cap-settings-section"><h3>Generation overrides</h3><p>Leave fields empty to use the server defaults.</p></div>
                <div class="cap-settings-grid">
                <div class="cap-field">
                    <label for="capTemperature">Temperature</label>
                    <input class="cap-input" id="capTemperature" type="number" min="0" max="2" step="0.05" placeholder="LM Studio default">
                </div>
                <div class="cap-field">
                    <label for="capTopP">Top P</label>
                    <input class="cap-input" id="capTopP" type="number" min="0" max="1" step="0.01" placeholder="LM Studio default">
                </div>
                <div class="cap-field">
                    <label for="capMaxTokens">Max tokens</label>
                    <input class="cap-input" id="capMaxTokens" type="number" min="1" max="131072" step="1" placeholder="LM Studio default">
                </div>
                <div class="cap-field">
                    <label for="capTopK">Top K</label>
                    <input class="cap-input" id="capTopK" type="number" min="0" max="1000" step="1" placeholder="LM Studio default">
                </div>
                <div class="cap-field">
                    <label for="capPresencePenalty">Presence penalty</label>
                    <input class="cap-input" id="capPresencePenalty" type="number" min="-2" max="2" step="0.1" placeholder="LM Studio default">
                </div>
                </div>
            <button class="cap-btn" type="button" id="clearCaptionerOverridesBtn" onclick="clearCaptionerOverrides()">Clear generation overrides</button>
            <p class="cap-settings-feedback" id="connectionFeedback" role="status"></p>
        </div>
        <footer class="cap-dialog-footer"><button class="cap-btn" type="button" id="detectModelBtn" onclick="detectCaptionerModel()">Detect model</button><div><button class="cap-btn" type="button" onclick="closeConnectionSettings()">Cancel</button><button class="cap-btn primary" id="saveConnectionBtn" type="submit">Save settings</button></div></footer>
    </form>
</dialog>
</div>

<div class="cap-toast" id="toast"></div>

<script>
const state = {
  images: [],
  activeIdx: null,
  running: false,
  stopRequested: false,
  activeRequestController: null,
  config: null,
  /* Loaded captioner presets (type=captioner cards from the Library).
     activePreset is the selected built-in script or saved card object.
     activeOverride is the live textarea content shipped as override_prompt; lets the user
     tweak before running without permanently changing the preset card. */
  presets: [],
  importedPreset: null,
  importingPrompt: false,
  presetBusy: false,
  activePreset: null,
  activeOverride: '',
  autoTagger: { available: false, ready: false, runtimeReady: false },
  saveSidecars: true,
  sidecarDirectory: null,
  sidecarDirectoryName: ''
};

const BUILTIN_VISION_SCRIPTS = [
  {
    id: 'built:z_image',
    title: 'Z-Image LoRA training',
    content: `You are a captioning assistant for Z-Image LoRA training datasets.
Analyze the image and write one flowing natural-language caption.

Begin with the trigger token: {TRIGGER}

Describe only the variable elements: pose, expression, clothing, action, environment, background, framing, shot distance, angle, lighting direction and quality.

Do not describe traits that stay constant across the dataset. Those must be absorbed by the trigger token.

Caption only what is clearly visible. Write in plain declarative sentences. Keep it between 40 and 70 words.
No names, no celebrity identities, no invented context, no quality markers, no weighting syntax, no watermark or text disclaimers.
Output only the caption text.`
  },
  {
    id: 'built:krea2_character',
    title: 'Krea 2 Character LoRA',
    content: `You are a captioning assistant for Krea 2 character and likeness LoRA training datasets.

Analyze the image and write one concise, flowing natural-language caption suitable for Krea 2's Qwen3-VL text encoder.

Begin with the trigger token: {TRIGGER}

Treat the trigger token as the identity of the recurring person or character.

After the trigger token, identify the subject with a generic class when appropriate, such as an adult woman or an adult man. Describe only the variable visible elements: pose, facial expression, clothing, accessories, hairstyle when it differs from the character's normal appearance, action, environment, background, framing, shot distance, camera angle, and clearly visible lighting conditions.

Do not repeatedly describe stable identity-defining traits that remain consistent across the dataset. The trigger token should learn the person's face, facial proportions, normal hair color, normal eye appearance, body proportions, and other persistent likeness characteristics.

Describe a normally stable trait only when it is visibly altered in that particular image and the difference is relevant, such as different hair styling, glasses, makeup, facial hair, or a temporary appearance change.

Do not describe the artistic style or rendering technique unless style variation itself is important to the dataset.

Caption only what is clearly visible. Do not infer personality, biography, relationships, intent, or story context.

Use natural descriptive sentences, never comma-separated tag lists. Keep the caption concise, normally between 25 and 60 words.

Do not use the person's real name, celebrity identity, artist names, quality markers, prompt weighting, aesthetic praise, or watermark and text disclaimers.

Output only the caption.`
  },
  {
    id: 'built:krea2_style',
    title: 'Krea 2 Style LoRA',
    content: `You are a captioning assistant for Krea 2 LoRA training datasets.

Analyze the image and write one concise, flowing natural-language caption suitable for Krea 2's Qwen3-VL text encoder.

Begin with the trigger token: {TRIGGER}

Treat the trigger token as the holder of the shared visual style. Describe the content of the image, not the style itself.

Identify the main subject generically, such as an adult woman, an adult man, two people, a creature, or an object. Describe the variable visible elements: pose, expression, clothing, accessories, action, environment, important background elements, framing, shot distance, camera angle, and clearly visible lighting conditions.

Do not describe visual traits that define the shared style. Do not mention the medium, artistic technique, rendering method, linework, brushwork, shading style, color treatment, illustration style, realism level, or other characteristics that remain consistent across the dataset. These must be learned through the trigger token.

Do not describe recurring subject traits as part of the style when they can be identified separately. If an adult woman is visible, explicitly describe her as an adult woman. Likewise identify other subjects generically so that subject matter is separated from the style trigger.

Caption only what is clearly visible. Do not infer hidden details, intent, story, or context.

Use natural descriptive sentences, never comma-separated tag lists. Keep the caption concise, normally between 25 and 60 words.

Do not use names, artist names, celebrity identities, quality markers, prompt weighting, aesthetic praise, or watermark and text disclaimers.

Output only the caption.`
  },
  {
    id: 'built:flux',
    title: 'Flux LoRA training',
    content: `You are a captioning assistant for Flux LoRA training datasets.
Analyze the image and write one natural-language caption paragraph.

Begin with the trigger token: {TRIGGER}

Describe only what varies across the dataset: pose, expression, clothing, action, setting, background elements, camera distance, angle, lighting and color palette.

Do not describe inherent traits of the subject that remain constant. Those belong to the trigger token alone.

Write concrete declarative prose. No hidden context, no names, no celebrity identities, no invented events, no unseen text or logos, no quality hype, no weighting syntax.
Keep it between 50 and 80 words.
Output one paragraph only.`
  },
  {
    id: 'built:sdxl',
    title: 'SDXL LoRA training',
    content: `You are a captioning assistant for LoRA training datasets (SDXL).
Analyze the image and write one comma-separated caption.

Begin every caption with the trigger token: {TRIGGER}

After the trigger, describe only the variable elements: pose, expression, clothing, background, setting, lighting, camera framing, composition and image quality.

Do not describe the inherent traits of the subject that stay constant across the dataset (face shape, hair color, eye color, body type, permanent markings). These must be absorbed by the trigger token.

Keep captions under 60 tokens. No invented details, no proper names, no celebrity identities, no hype filler, no weighted syntax, no duplicate tags.
Output only the final comma-separated caption.`
  },
  {
    id: 'built:sdxl_wd_hybrid',
    title: 'SDXL WD Hybrid',
    requiresAutoTagger: true,
    hybrid: true,
    content: `You are a caption refiner for SDXL LoRA training datasets.

You receive an image, a trigger token, a filtered WD tag list, and a list of constant subject traits.

Return one comma-separated caption using concise booru-style phrases.

Start with the exact trigger token: {TRIGGER}

Keep relevant WD tags describing variable elements such as pose, expression, clothing, action, background, framing and subject count.

Remove tags listed in CONSTANT_TRAITS. Do not remove other subject tags merely because they might be constant.

Add only clearly visible details that the WD tags missed: lighting, materials, textures, spatial relationships, camera angle, shot distance and color palette.

Do not add rating, character identity, meta or quality tags.
No invented details, names, celebrity identities, weighting syntax or duplicates.
Use normal spaces inside phrases, never underscores. Keep the trigger token unchanged.
Use at most 45 tags.

WD_TAGS:
{WD_TAGS}

CONSTANT_TRAITS:
{CONSTANT_TRAITS}

Output only the final comma-separated caption.`
  },
  {
    id: 'built:dataset',
    title: 'Dataset training caption',
    content: `You are preparing concise dataset captions for image model training.

Describe the image with stable, reusable visual terms. Focus on subject identity as a category, clothing, pose, framing, environment, style, lighting and notable objects. Prefer clear nouns and adjectives over poetic language.

Do not include guesses, story context, emotions that are not visible, quality ratings, camera EXIF, watermarks, logos or unreadable text.

Output one concise caption only.`
  },
  {
    id: 'built:plain',
    title: 'Plain visual description',
    content: `Describe the image accurately in plain language.

Mention only visible subjects, objects, setting, composition, lighting, colors and style. Keep uncertain details out. Do not identify real people, brands or copyrighted characters unless text in the image clearly names them.

Output one clear paragraph only.`
  }
];

const SESSION_DB = 'cyberdelia-captioner-session';
const SESSION_STORE = 'session';
const SESSION_KEY = 'current';
const TRIGGER_STORAGE_KEY = 'cyberdelia.captioner.triggerWord';
const SIDECAR_STORAGE_KEY = 'cyberdelia.captioner.saveSidecars';
const CONSTANT_TRAITS_STORAGE_KEY = 'cyberdelia.captioner.constantTraits';
const WD_THRESHOLD_STORAGE_KEY = 'cyberdelia.captioner.wdThreshold';
let sessionSaveTimer = null;
let restoringSession = false;
let sessionSaveState = '';
let sessionSaveRevision = 0;

async function init() {
  setupConnectionDialog();
  document.getElementById('presetTitle').addEventListener('input', () => {
    syncImportedPrompt();
    scheduleSaveSession();
  });
  await loadConnectionConfig();
  checkConnection();
  setInterval(checkConnection, 15000);
  await loadAutoTaggerStatus();
  await loadPresets();
  await restoreSession();
  /* Track edits in the live textarea so each caption picks up the current text */
  const ed = document.getElementById('presetEditor');
  if (ed) ed.addEventListener('input', function(){
    state.activeOverride = ed.value;
    syncImportedPrompt();
    updateWdHybridUi();
    scheduleSaveSession();
  });
  const targetSel = document.getElementById('saveTargetSelect');
  if (targetSel) targetSel.addEventListener('change', scheduleSaveSession);
  const triggerInput = document.getElementById('capTriggerWord');
  if (triggerInput) {
    triggerInput.value = localStorage.getItem(TRIGGER_STORAGE_KEY) || '';
    triggerInput.addEventListener('input', function(){
      localStorage.setItem(TRIGGER_STORAGE_KEY, triggerInput.value);
    });
  }
  const constantTraits = document.getElementById('capConstantTraits');
  if (constantTraits) {
    constantTraits.value = localStorage.getItem(CONSTANT_TRAITS_STORAGE_KEY) || '';
    constantTraits.addEventListener('input', function(){
      localStorage.setItem(CONSTANT_TRAITS_STORAGE_KEY, constantTraits.value);
    });
  }
  const wdThreshold = document.getElementById('capWdThreshold');
  if (wdThreshold) {
    wdThreshold.value = localStorage.getItem(WD_THRESHOLD_STORAGE_KEY) || '0.40';
    wdThreshold.addEventListener('change', function(){
      const value = Math.max(0.05, Math.min(0.95, Number(wdThreshold.value) || 0.40));
      wdThreshold.value = value.toFixed(2);
      localStorage.setItem(WD_THRESHOLD_STORAGE_KEY, wdThreshold.value);
    });
  }
  updateWdHybridUi();
  const sidecarToggle = document.getElementById('saveSidecarsToggle');
  if (sidecarToggle) {
    state.saveSidecars = localStorage.getItem(SIDECAR_STORAGE_KEY) !== '0';
    sidecarToggle.checked = state.saveSidecars;
    sidecarToggle.addEventListener('change', function(){
      state.saveSidecars = sidecarToggle.checked;
      localStorage.setItem(SIDECAR_STORAGE_KEY, state.saveSidecars ? '1' : '0');
      updateSidecarUi();
    });
  }
  updateSidecarUi();
}

function populateConnectionForm(cfg) {
  document.getElementById('capApiUrl').value = cfg.api_url || 'http://localhost:1234/v1';
  document.getElementById('capModel').value = cfg.model || '';
  for (const [key, id] of Object.entries({ temperature: 'capTemperature', top_p: 'capTopP', max_tokens: 'capMaxTokens', top_k: 'capTopK', presence_penalty: 'capPresencePenalty' })) {
    document.getElementById(id).value = cfg[key] ?? '';
  }
}

function connectionFeedback(message='', error=false) {
  const el = document.getElementById('connectionFeedback');
  el.textContent = message;
  el.dataset.error = String(error);
}

function setupConnectionDialog() {
  const dialog = document.getElementById('connectionDialog');
  dialog.addEventListener('click', event => {
    const rect = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) closeConnectionSettings();
  });
  dialog.addEventListener('cancel', event => {
    if (state.connectionBusy) event.preventDefault();
  });
  dialog.addEventListener('close', () => populateConnectionForm(state.config || {}));
}

function openConnectionSettings() {
  populateConnectionForm(state.config || {});
  connectionFeedback();
  document.getElementById('connectionDialog').showModal();
  document.getElementById('capApiUrl').focus();
}

function closeConnectionSettings() {
  if (!state.connectionBusy) document.getElementById('connectionDialog').close();
}

function setConnectionBusy(busy) {
  state.connectionBusy = busy;
  document.querySelectorAll('#connectionForm button, #connectionForm input').forEach(el => { el.disabled = busy; });
}

async function loadConnectionConfig() {
  try {
    const r = await fetch('/api/captioner/config');
    if (!r.ok) throw new Error('Config unavailable');
    const cfg = await r.json();
    state.config = cfg || {};
    populateConnectionForm(state.config);
  } catch(e) {
    toast('Could not load captioner settings', 'error-toast');
  }
}

function normalizeCaptionerApiUrl(value) {
  let url = String(value || '').trim() || 'http://localhost:1234';
  if (!/^https?:\/\//i.test(url)) url = 'http://' + url;
  url = url.replace(/\/+$/, '');
  if (!/\/v1$/i.test(url)) url += '/v1';
  return url;
}

function optionalFormNumber(id, integer=false) {
  const raw = String(document.getElementById(id).value || '').trim();
  if (!raw) return null;
  const value = integer ? parseInt(raw, 10) : Number(raw);
  return Number.isFinite(value) ? value : null;
}

function readConnectionForm() {
  const apiUrl = normalizeCaptionerApiUrl(document.getElementById('capApiUrl').value);
  document.getElementById('capApiUrl').value = apiUrl;
  return {
    api_url: apiUrl,
    model: (document.getElementById('capModel').value || '').trim(),
    temperature: optionalFormNumber('capTemperature'),
    top_p: optionalFormNumber('capTopP'),
    max_tokens: optionalFormNumber('capMaxTokens', true),
    top_k: optionalFormNumber('capTopK', true),
    presence_penalty: optionalFormNumber('capPresencePenalty')
  };
}

function clearCaptionerOverrides() {
  ['capTemperature', 'capTopP', 'capMaxTokens', 'capTopK', 'capPresencePenalty']
    .forEach(id => { document.getElementById(id).value = ''; });
  connectionFeedback('Overrides cleared. Save settings to use the server defaults.');
}

async function saveConnectionConfig() {
  if (state.connectionBusy || !document.getElementById('connectionForm').reportValidity()) return false;
  const cfg = readConnectionForm();
  setConnectionBusy(true);
  connectionFeedback('Saving settings…');
  try {
    const r = await fetch('/api/captioner/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg)
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok || data.error) throw new Error(data.error || 'Save failed');
    state.config = cfg;
    document.getElementById('connectionDialog').close();
    toast('Captioner settings saved', 'success');
    checkConnection();
    return true;
  } catch(e) {
    connectionFeedback('Save failed: ' + e.message, true);
    return false;
  } finally {
    setConnectionBusy(false);
  }
}

async function detectCaptionerModel() {
  if (state.connectionBusy || !document.getElementById('connectionForm').reportValidity()) return;
  const cfg = readConnectionForm();
  setConnectionBusy(true);
  connectionFeedback('Looking for loaded models…');
  try {
    const data = await fetchCaptionerModels(cfg);
    document.getElementById('capModel').value = data.models[0];
    connectionFeedback('Detected ' + data.models[0] + '. Save settings to use this model.');
  } catch(e) {
    connectionFeedback('Detect failed: ' + e.message, true);
  } finally {
    setConnectionBusy(false);
    document.getElementById('detectModelBtn').focus();
  }
}

async function fetchCaptionerModels(cfg=state.config || {}) {
  let directError = null;
  try {
    const r = await fetch(normalizeCaptionerApiUrl(cfg.api_url) + '/models', { signal: AbortSignal.timeout(5000) });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error?.message || data.error || `HTTP ${r.status}`);
    const models = (data.data || []).map(item => item && item.id).filter(Boolean);
    if (!models.length) throw new Error('No models found');
    return { models, via: 'browser' };
  } catch(e) {
    directError = e;
  }

  if (normalizeCaptionerApiUrl(cfg.api_url) !== normalizeCaptionerApiUrl(state.config?.api_url)) {
    throw new Error('Save this API URL first to detect models through CyberHub.');
  }
  const r = await fetch('/api/captioner/models');
  const data = await r.json().catch(() => ({}));
  if (!r.ok || data.error || !(data.models || []).length) {
    throw new Error(data.error || directError?.message || 'No models found');
  }
  return { models: data.models, via: 'hub' };
}

/* Saved captioner presets are Library cards with type='captioner'. Built-in
   vision scripts are local defaults; save one as a preset before updating it. */
async function loadPresets(options={}) {
  renderPresetOptions();
  if (!state.activePreset) onPresetChange({ skipSave: true });
  try {
    const response = await fetch('/api/captioner/presets?_=' + Date.now(), { cache: 'no-store' });
    const data = await response.json();
    if (!response.ok || !Array.isArray(data.cards)) throw new Error(data.error || 'Could not load presets');
    state.presets = data.cards;
  } catch(e) { /* The local scripts and imported prompt remain available. */ }
  renderPresetOptions(options.selectedValue);
  if (options.selectedValue || !state.activePreset) onPresetChange({ skipSave: true });
}

function renderPresetOptions(selectedValue) {
  const sel = document.getElementById('presetSelect');
  const cur = selectedValue || sel.value || 'built:z_image';
  const visibleBuiltins = BUILTIN_VISION_SCRIPTS.filter(p => !p.requiresAutoTagger || state.autoTagger.available);
  const builtins = '<optgroup label="Built-in vision scripts">'
    + visibleBuiltins.map(p => '<option value="' + p.id + '">' + escapeHtml(p.title) + '</option>').join('')
    + '</optgroup>';
  const imported = state.importedPreset
    ? '<optgroup label="Imported prompt"><option value="imported:current">' + escapeHtml(state.importedPreset.title || 'Untitled import') + '</option></optgroup>'
    : '';
  const titleCounts = new Map();
  state.presets.forEach(p => {
    const key = String(p.title || '').normalize('NFKC').trim().replace(/\s+/g, ' ').toLowerCase();
    titleCounts.set(key, (titleCounts.get(key) || 0) + 1);
  });
  const saved = state.presets.length
    ? '<optgroup label="Saved presets">' + state.presets.map(p => {
        const key = String(p.title || '').normalize('NFKC').trim().replace(/\s+/g, ' ').toLowerCase();
        const label = (p.title || 'Untitled') + (titleCounts.get(key) > 1 ? ' · #' + p.id : '');
        return '<option value="saved:' + escapeHtml(p.id) + '">' + escapeHtml(label) + '</option>';
      }).join('') + '</optgroup>'
    : '';
  sel.innerHTML = builtins + imported + saved;
  sel.value = Array.from(sel.options).some(opt => opt.value === cur) ? cur : 'built:z_image';
}

function onPresetChange(options={}) {
  const sel = document.getElementById('presetSelect');
  const ed  = document.getElementById('presetEditor');
  const upd = document.getElementById('updatePresetBtn');
  const value = sel.value || 'built:z_image';
  const isSaved = value.startsWith('saved:');
  const rawId = isSaved ? value.slice(6) : value;
  const p = isSaved
    ? state.presets.find(function(x){ return String(x.id) === String(rawId); })
    : value === 'imported:current' ? state.importedPreset
    : BUILTIN_VISION_SCRIPTS.find(function(x){ return x.id === rawId; });
  if (!p) return;
  state.activePreset = Object.assign({ builtin: !isSaved }, p);
  state.activeOverride = p.content || '';
  document.getElementById('presetTitle').value = p.title || '';
  ed.value = state.activeOverride;
  ed.classList.add('active');
  upd.style.display = isSaved ? '' : 'none';
  document.getElementById('deletePresetBtn').style.display = isSaved ? '' : 'none';
  promptTransferFeedback(value === 'imported:current'
    ? 'Imported into the editor. Save as preset to add it to your Library.'
    : 'Share a title and system prompt as JSON.');
  updateWdHybridUi();
  if (!options.skipSave) scheduleSaveSession();
}

async function loadAutoTaggerStatus() {
  try {
    const r = await fetch('/api/auto_tagger/status');
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    state.autoTagger = {
      available: true,
      ready: !!data?.model?.ready,
      runtimeReady: !!data?.runtime?.ready,
      status: data
    };
  } catch(e) {
    state.autoTagger = {
      available: false,
      ready: false,
      runtimeReady: false,
      status: null
    };
  }
  updateWdHybridUi();
  return state.autoTagger;
}

function isWdHybridPreset() {
  return !!(state.activePreset?.hybrid || /\{WD_TAGS\}/i.test(state.activeOverride || ''));
}

function updateWdHybridUi() {
  const controls = document.getElementById('wdHybridControls');
  const status = document.getElementById('wdHybridStatus');
  if (!controls || !status) return;
  const active = isWdHybridPreset();
  controls.classList.toggle('active', active);
  if (!active) return;

  const autoTagger = state.autoTagger;
  if (!autoTagger.available) {
    status.innerHTML = 'Auto Tagger is not installed. Use the standard SDXL preset instead.';
    return;
  }
  if (!autoTagger.runtimeReady) {
    const message = autoTagger.status?.runtime?.message || 'Auto Tagger runtime is unavailable';
    status.innerHTML = escapeHtml(message) + ' · <a href="/auto_tagger">Open Auto Tagger</a>';
    return;
  }
  if (!autoTagger.ready) {
    status.innerHTML = 'Install WD EVA02-Large first · <a href="/auto_tagger">Open Auto Tagger</a>';
    return;
  }
  const runtime = autoTagger.status?.runtime?.message || 'local runtime';
  status.textContent = `WD EVA02-Large ready · ${runtime}`;
}

function openSessionDb() {
  return new Promise((resolve, reject) => {
    if (!('indexedDB' in window)) { reject(new Error('IndexedDB unavailable')); return; }
    const req = indexedDB.open(SESSION_DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(SESSION_STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error || new Error('IndexedDB failed'));
  });
}

async function sessionStore(mode, fn) {
  const db = await openSessionDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(SESSION_STORE, mode);
    const store = tx.objectStore(SESSION_STORE);
    let result;
    try { result = fn(store); }
    catch(e) { db.close(); reject(e); return; }
    tx.oncomplete = () => { db.close(); resolve(result && result.result !== undefined ? result.result : result); };
    tx.onerror = () => { db.close(); reject(tx.error || new Error('Session store failed')); };
  });
}

function normalizeSessionStatus(status) {
  return status === 'running' ? 'pending' : (status || 'pending');
}

function scheduleSaveSession() {
  if (restoringSession) return;
  clearTimeout(sessionSaveTimer);
  sessionSaveRevision++;
  sessionSaveState = 'saving';
  updateCaptionSaveStatus();
  sessionSaveTimer = setTimeout(saveSessionNow, 300);
}

async function saveSessionNow() {
  if (restoringSession) return;
  const revision = sessionSaveRevision;
  const presetSel = document.getElementById('presetSelect');
  const targetSel = document.getElementById('saveTargetSelect');
  const record = {
    saved_at: Date.now(),
    activeIdx: state.activeIdx,
    activeOverride: state.activeOverride || '',
    presetTitle: document.getElementById('presetTitle').value,
    importedPreset: state.importedPreset,
    presetId: presetSel ? presetSel.value : '',
    target: targetSel ? targetSel.value : 'general',
    images: state.images.map(img => ({
      id: img.id,
      file: img.file,
      name: img.name,
      size: img.size,
      caption: img.caption || '',
      status: normalizeSessionStatus(img.status),
      sidecarStatus: img.sidecarStatus || ''
    }))
  };
  try {
    await sessionStore('readwrite', store => store.put(record, SESSION_KEY));
    if (revision === sessionSaveRevision) sessionSaveState = 'saved';
  } catch(e) {
    if (revision === sessionSaveRevision) sessionSaveState = 'error';
    console.warn('Captioner session was not saved:', e);
  }
  updateCaptionSaveStatus();
}

async function clearSession() {
  clearTimeout(sessionSaveTimer);
  try {
    await sessionStore('readwrite', store => store.delete(SESSION_KEY));
  } catch(e) {
    console.warn('Captioner session was not cleared:', e);
  }
}

async function restoreSession() {
  restoringSession = true;
  try {
    const record = await sessionStore('readonly', store => store.get(SESSION_KEY));
    if (!record || !Array.isArray(record.images)) return;

    state.images.forEach(i => i.url && URL.revokeObjectURL(i.url));
    state.images = record.images
      .filter(item => item && item.file)
      .map(item => {
        const file = item.file instanceof File
          ? item.file
          : new File([item.file], item.name || 'image', { type: item.file.type || 'image/jpeg' });
        return {
          id: item.id || (Date.now() + Math.random()),
          file: file,
          name: item.name || file.name,
          size: item.size || file.size,
          url: URL.createObjectURL(file),
          caption: item.caption || '',
          status: normalizeSessionStatus(item.status),
          sidecarStatus: item.sidecarStatus || ''
        };
      });

    if (record.importedPreset && typeof record.importedPreset.title === 'string'
        && typeof record.importedPreset.content === 'string') {
      state.importedPreset = {id: 'imported:current', title: record.importedPreset.title, content: record.importedPreset.content};
    }
    const presetSel = document.getElementById('presetSelect');
    if (presetSel && record.presetId) {
      const restoredPresetId = String(record.presetId).startsWith('built:') || String(record.presetId).startsWith('saved:') || record.presetId === 'imported:current'
        ? String(record.presetId)
        : 'saved:' + record.presetId;
      renderPresetOptions(restoredPresetId);
      onPresetChange({ skipSave: true });
    }
    if (typeof record.activeOverride === 'string') {
      state.activeOverride = record.activeOverride;
      const ed = document.getElementById('presetEditor');
      if (ed) {
        ed.value = record.activeOverride;
        if (record.presetId) ed.classList.add('active');
      }
    }
    if (typeof record.presetTitle === 'string') document.getElementById('presetTitle').value = record.presetTitle;
    syncImportedPrompt();
    updateWdHybridUi();
    const targetSel = document.getElementById('saveTargetSelect');
    if (targetSel && record.target) targetSel.value = record.target;

    renderQueue();
    if (state.images.length) {
      const idx = Number.isInteger(record.activeIdx) && record.activeIdx >= 0 && record.activeIdx < state.images.length
        ? record.activeIdx
        : 0;
      selectImage(idx, { skipSave: true });
      toast('Restored previous Captioner batch', 'success');
    }
    updateControls();
  } catch(e) {
    console.warn('Captioner session was not restored:', e);
  } finally {
    restoringSession = false;
  }
}

// Versioned envelope shared by CyberHub system-prompt files. Each module owns
// its payload; never infer the destination from a filename or a Library card type.
const PROMPT_SHARE_SCHEMA = 1;
const PROMPT_SHARE_MAX_BYTES = 512 * 1024;
const PROMPT_SHARE_MAX_TITLE = 200;
const PROMPT_SHARE_MAX_TEXT = 100000;

function validatePromptShare(data) {
  if (!data || typeof data !== 'object' || Array.isArray(data)
      || data.product !== 'CyberHub' || data.type !== 'system_prompt') {
    throw new Error('Choose a CyberHub system-prompt JSON file.');
  }
  if (data.module !== 'captioner') {
    throw new Error('This prompt is for another module. Choose a Captioner prompt.');
  }
  if (data.schema !== PROMPT_SHARE_SCHEMA) {
    throw new Error('Unsupported prompt format version. Update Captioner or use schema 1.');
  }
  if (typeof data.title !== 'string' || !data.title.trim() || data.title.trim().length > PROMPT_SHARE_MAX_TITLE) {
    throw new Error('The prompt needs a title of 1–200 characters.');
  }
  if (typeof data.prompt !== 'string' || !data.prompt.trim() || data.prompt.length > PROMPT_SHARE_MAX_TEXT) {
    throw new Error('The system prompt must contain text and be at most 100,000 characters.');
  }
  return { schema: PROMPT_SHARE_SCHEMA, product: 'CyberHub', type: 'system_prompt',
    module: 'captioner', title: data.title.trim(), prompt: data.prompt };
}

function currentPromptShare() {
  return validatePromptShare({ schema: PROMPT_SHARE_SCHEMA, product: 'CyberHub',
    type: 'system_prompt', module: 'captioner',
    title: document.getElementById('presetTitle').value,
    prompt: document.getElementById('presetEditor').value });
}

function promptTransferFeedback(message, error=false) {
  const el = document.getElementById('promptTransferStatus');
  el.textContent = message;
  el.dataset.error = String(error);
}

function syncImportedPrompt() {
  if (document.getElementById('presetSelect').value !== 'imported:current' || !state.importedPreset) return;
  state.importedPreset.title = document.getElementById('presetTitle').value;
  state.importedPreset.content = document.getElementById('presetEditor').value;
  renderPresetOptions('imported:current');
}

function applyImportedPrompt(data) {
  const prompt = validatePromptShare(data);
  state.importedPreset = {id: 'imported:current', title: prompt.title, content: prompt.prompt};
  renderPresetOptions('imported:current');
  onPresetChange();
}

async function importPromptFile(input) {
  const file = input.files?.[0];
  if (!file || state.importingPrompt || state.presetBusy) return;
  state.importingPrompt = true;
  document.getElementById('importPromptBtn').disabled = true;
  try {
    if (file.size > PROMPT_SHARE_MAX_BYTES) throw new Error('Prompt JSON files must be smaller than 512 KB.');
    const text = (await file.text()).replace(/^\uFEFF/, '');
    let data;
    try { data = JSON.parse(text); }
    catch { throw new Error('This file is not valid JSON.'); }
    applyImportedPrompt(data);
    toast('Captioner prompt imported', 'success');
  } catch(e) {
    promptTransferFeedback('Import failed: ' + e.message, true);
  } finally {
    input.value = '';
    state.importingPrompt = false;
    document.getElementById('importPromptBtn').disabled = false;
  }
}

function exportPromptJson() {
  try {
    const data = currentPromptShare();
    const json = JSON.stringify(data, null, 2) + '\n';
    if (new Blob([json]).size > PROMPT_SHARE_MAX_BYTES) throw new Error('This prompt is too large to share as JSON (512 KB maximum).');
    const stem = data.title.normalize('NFKD').replace(/[\u0300-\u036f]/g, '')
      .toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 80) || 'system-prompt';
    download(stem + '.captioner.json', json, 'application/json');
    promptTransferFeedback('Exported the current title and prompt. Ready to share.');
  } catch(e) {
    promptTransferFeedback('Export failed: ' + e.message, true);
  }
}

function setPresetBusy(busy) {
  state.presetBusy = busy;
  ['savePresetBtn', 'updatePresetBtn', 'deletePresetBtn', 'importPromptBtn',
   'exportPromptBtn', 'presetSelect', 'presetTitle', 'presetEditor'].forEach(id => {
    document.getElementById(id).disabled = busy;
  });
}

async function storePreset(preset=null) {
  if (state.presetBusy || state.importingPrompt) return;
  let shared;
  try { shared = currentPromptShare(); }
  catch(e) { promptTransferFeedback(e.message, true); return; }
  if (preset && !confirm('Update saved preset "' + preset.title + '" with the current title and prompt?')) return;
  setPresetBusy(true);
  promptTransferFeedback('Saving preset…');
  try {
    const payload = {title: shared.title, content: shared.prompt};
    if (preset) payload.id = preset.id;
    const response = await fetch('/api/captioner/presets/save', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || 'Could not save preset');
    // Update immediately from the committed server result, even if a later
    // refresh fails. This also makes repeated Save clicks see the new preset.
    state.presets = state.presets.filter(p => String(p.id) !== String(data.id));
    state.presets.unshift(data.preset);
    renderPresetOptions('saved:' + data.id);
    onPresetChange();
    promptTransferFeedback(preset ? 'Preset updated in the Library.' : 'Preset saved in the Library. Use Update to change it.');
    toast(preset ? 'Preset updated' : 'Preset saved', 'success');
  } catch(e) {
    promptTransferFeedback('Save failed: ' + e.message, true);
  } finally {
    setPresetBusy(false);
  }
}

async function saveAsPreset() {
  return storePreset();
}

async function updatePreset() {
  if (!state.activePreset || state.activePreset.builtin) return;
  return storePreset(state.activePreset);
}

async function deletePreset() {
  if (state.presetBusy || state.importingPrompt || !state.activePreset || state.activePreset.builtin) return;
  const preset = state.activePreset;
  if (!confirm('Delete preset "' + preset.title + '" (#' + preset.id + ') from the Library? This cannot be undone.')) return;
  setPresetBusy(true);
  promptTransferFeedback('Deleting preset…');
  try {
    const response = await fetch('/api/captioner/presets/delete', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({id: preset.id})
    });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || 'Could not delete preset');
    state.presets = state.presets.filter(p => String(p.id) !== String(preset.id));
    renderPresetOptions('built:z_image');
    onPresetChange();
    promptTransferFeedback('Preset deleted from the Library.');
    toast('Preset deleted', 'success');
  } catch(e) {
    promptTransferFeedback('Delete failed: ' + e.message, true);
  } finally {
    setPresetBusy(false);
  }
}

async function checkConnection() {
  const dot = document.getElementById('statusDot');
  const lbl = document.getElementById('statusLabel');
  try {
    const data = await fetchCaptionerModels();
    const count = data.models.length;
    dot.className = 'cap-status-dot connected';
    lbl.textContent = `Connected · ${count} model${count === 1 ? '' : 's'}`;
  } catch {
    dot.className = 'cap-status-dot error';
    lbl.textContent = 'Not connected · open settings';
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const folderInput = document.getElementById('folderInput');

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  addFiles(Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/')));
});
fileInput.addEventListener('change', () => {
  addFiles(Array.from(fileInput.files));
  fileInput.value = '';
});

folderInput.addEventListener('change', () => {
  const files = Array.from(folderInput.files).filter(isCaptionerImageFile);
  addFiles(files);
  folderInput.value = '';
  if (files.length) {
    toast(`Added ${files.length} images. Use Save caption files to download matching .txt files.`);
  }
});

function isCaptionerImageFile(file) {
  return !!file && (String(file.type || '').startsWith('image/') ||
    /\.(png|jpe?g|webp|bmp|gif|tiff?)$/i.test(file.name || ''));
}

function supportsWritableFolders() {
  return window.isSecureContext && typeof window.showDirectoryPicker === 'function';
}

function setSidecarsEnabled(enabled) {
  state.saveSidecars = !!enabled;
  const toggle = document.getElementById('saveSidecarsToggle');
  if (toggle) toggle.checked = state.saveSidecars;
  localStorage.setItem(SIDECAR_STORAGE_KEY, state.saveSidecars ? '1' : '0');
  updateSidecarUi();
}

function updateSidecarUi() {
  const status = document.getElementById('sidecarFolderStatus');
  if (!status) return;
  if (!state.saveSidecars) {
    status.textContent = 'Automatic caption file saving is off.';
  } else if (state.sidecarDirectory) {
    status.textContent = `Caption files: ${state.sidecarDirectoryName || 'selected folder'}`;
    status.title = status.textContent;
  } else if (supportsWritableFolders()) {
    status.textContent = 'The image folder is requested when captioning starts.';
  } else {
    status.textContent = 'Direct folder writing is unavailable here; Save caption files downloads a ZIP.';
  }
}

async function requestSidecarPermission(handle) {
  if (!handle) return false;
  if (typeof handle.queryPermission !== 'function') return true;
  let permission = await handle.queryPermission({ mode: 'readwrite' });
  if (permission !== 'granted' && typeof handle.requestPermission === 'function') {
    permission = await handle.requestPermission({ mode: 'readwrite' });
  }
  return permission === 'granted';
}

async function chooseSidecarDirectory() {
  if (!supportsWritableFolders()) return null;
  const handle = await window.showDirectoryPicker({ mode: 'readwrite' });
  if (!await requestSidecarPermission(handle)) throw new Error('Write access was not granted');
  state.sidecarDirectory = handle;
  state.sidecarDirectoryName = handle.name || '';
  setSidecarsEnabled(true);
  return handle;
}

async function openImageFolder() {
  if (!supportsWritableFolders()) {
    folderInput.click();
    return;
  }
  try {
    const handle = await chooseSidecarDirectory();
    const files = [];
    for await (const entry of handle.values()) {
      if (entry.kind !== 'file' || !isCaptionerImageFile({ name: entry.name, type: '' })) continue;
      files.push(await entry.getFile());
    }
    files.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
    if (!files.length) return toast('No supported images found in that folder', 'error-toast');
    addFiles(files);
    toast(`Added ${files.length} images; .txt caption files will be saved automatically`, 'success');
  } catch(e) {
    if (e && e.name === 'AbortError') return;
    toast(`Could not open folder: ${e.message}`, 'error-toast');
  }
}

async function prepareSidecarDirectory() {
  if (!state.saveSidecars) return false;
  try {
    if (state.sidecarDirectory && await requestSidecarPermission(state.sidecarDirectory)) return true;
    state.sidecarDirectory = null;
    state.sidecarDirectoryName = '';
    if (!supportsWritableFolders()) {
      setSidecarsEnabled(false);
      toast('Automatic caption files need localhost folder access. Use Save caption files for a ZIP.', 'error-toast');
      return false;
    }
    return !!await chooseSidecarDirectory();
  } catch(e) {
    if (e && e.name === 'AbortError') {
      setSidecarsEnabled(false);
      toast('Captioning continues without automatic caption files');
      return false;
    }
    setSidecarsEnabled(false);
    toast(`Caption folder unavailable: ${e.message}`, 'error-toast');
    return false;
  }
}

function sidecarFilename(imageName) {
  const name = String(imageName || 'caption');
  const dot = name.lastIndexOf('.');
  return (dot > 0 ? name.slice(0, dot) : name) + '.txt';
}

async function writeSidecar(img) {
  if (!state.sidecarDirectory || !img) return false;
  const directory = state.sidecarDirectory;
  const write = async () => {
    const caption = String(img.caption || '');
    const revision = img.editRevision || 0;
    const fileHandle = await directory.getFileHandle(sidecarFilename(img.name), { create: true });
    const writable = await fileHandle.createWritable();
    try {
      await writable.write(caption.trim() + '\n');
    } finally {
      await writable.close();
    }
    img.sidecarStatus = img.caption === caption && (img.editRevision || 0) === revision ? 'saved' : 'pending';
    return true;
  };
  const pending = (img.sidecarWrite || Promise.resolve()).catch(() => {}).then(write);
  img.sidecarWrite = pending;
  try { return await pending; }
  finally { if (img.sidecarWrite === pending) delete img.sidecarWrite; }
}

function addFiles(files) {
  let added = 0;
  let skipped = 0;
  files.forEach(file => {
    const existing = state.images.find(i => i.name === file.name && i.size === file.size);
    if (existing) { skipped++; return; }
    const id = Date.now() + Math.random();
    const url = URL.createObjectURL(file);
    state.images.push({ id, file, name: file.name, size: file.size, url, caption: '', status: 'pending' });
    added++;
  });
  renderQueue();
  if (state.activeIdx === null && state.images.length > 0) selectImage(0);
  updateControls();
  if (added > 0) scheduleSaveSession();
  if (added === 0 && skipped > 0) toast(`Skipped ${skipped} duplicate${skipped === 1 ? '' : 's'}`, 'error-toast');
  else if (skipped > 0) toast(`Added ${added}, skipped ${skipped} duplicate${skipped === 1 ? '' : 's'}`);
}

function renderQueue() {
  const list = document.getElementById('queueList');
  const focusedImageId = document.activeElement?.closest?.('.cap-queue-item')?.dataset.imageId;
  list.innerHTML = state.images.length ? '' : '<div class="cap-queue-empty">Your images will appear here.<br>Add files or open a folder to start.</div>';
  document.getElementById('queueCount').textContent = state.images.length;
  state.images.forEach((img, idx) => {
    const safeName = escapeHtml(img.name);
    const statusClass = img.sidecarStatus === 'error' ? 'error' :
      img.sidecarStatus === 'pending' && img.status !== 'running' ? 'edited' : img.status;
    const statusText = img.status === 'pending' ? 'pending' :
      img.status === 'running' ? '<span class="cap-spinner"></span>' + escapeHtml(img.stage || 'captioning') :
      img.sidecarStatus === 'pending' ? 'Edited · save .txt' :
      img.status === 'done' && img.sidecarStatus === 'saved' ? '✓ done · .txt' :
      img.status === 'done' && img.sidecarStatus === 'error' ? '✓ done · .txt failed' :
      img.status === 'done' ? '✓ done' : '✗ error';
    const item = document.createElement('button');
    item.type = 'button';
    item.title = img.name;
    item.dataset.imageId = String(img.id);
    item.setAttribute('aria-pressed', String(idx === state.activeIdx));
    item.className = 'cap-queue-item' + (idx === state.activeIdx ? ' active' : '');
    item.onclick = () => selectImage(idx);
    item.innerHTML = `
      <img class="cap-queue-thumb" src="${img.url}" alt="">
      <div class="cap-queue-info">
        <div class="cap-queue-name">${safeName}</div>
        <div class="cap-queue-status ${statusClass}">${statusText}</div>
      </div>
    `;
    list.appendChild(item);
    if (item.dataset.imageId === focusedImageId) item.focus({ preventScroll: true });
  });
  updateProgress();
}

function selectImage(idx, options={}) {
  state.activeIdx = idx;
  const img = state.images[idx];
  renderQueue();
  const panel = document.getElementById('imagePanel');
  panel.innerHTML = `<img src="${img.url}" alt="${escapeHtml(img.name)}">`;
  document.getElementById('captionOutput').value = img.caption || '';
  document.getElementById('currentFile').textContent = img.name;
  updateTokenCount();
  updateControls();
  if (!options.skipSave) scheduleSaveSession();
}

function updateProgress() {
  const total = state.images.length;
  const done = state.images.filter(i => i.status === 'done').length;
  const pct = total ? (done / total * 100) : 0;
  document.getElementById('progressFill').style.width = pct + '%';
  document.getElementById('progressText').textContent = total
    ? `${done} / ${total} captioned`
    : 'no images loaded';
}

function updateControls() {
  const has = state.images.length > 0;
  const hasUncaptioned = state.images.some(img => !String(img.caption || '').trim());
  const runAllBtn = document.getElementById('runAllBtn');
  const runUncaptionedBtn = document.getElementById('runUncaptionedBtn');
  const stopAllBtn = document.getElementById('stopAllBtn');
  runAllBtn.disabled = !has || state.running;
  runAllBtn.style.display = state.running ? 'none' : '';
  runUncaptionedBtn.disabled = !hasUncaptioned || state.running;
  runUncaptionedBtn.style.display = state.running ? 'none' : '';
  stopAllBtn.style.display = state.running ? '' : 'none';
  stopAllBtn.disabled = !state.running || state.stopRequested;
  stopAllBtn.textContent = state.stopRequested ? 'Stopping…' : 'Stop';
  document.getElementById('clearBtn').disabled = !has || state.running;
  document.getElementById('runOneBtn').disabled = !has || state.activeIdx === null || state.running;
  document.getElementById('captionOutput').disabled = state.activeIdx === null;
  document.getElementById('saveCaptionFilesBtn').disabled = !state.images.some(img => String(img.caption || '').trim());
  updateCaptionSaveStatus();
}

function updateTokenCount() {
  const txt = document.getElementById('captionOutput').value;
  const words = txt.trim() ? txt.trim().split(/\s+/).length : 0;
  document.getElementById('tokenCount').textContent = `~${words} words`;
}

function updateCaptionSaveStatus() {
  const el = document.getElementById('captionSaveStatus');
  const img = state.images[state.activeIdx];
  el.dataset.error = String(sessionSaveState === 'error' || img?.sidecarStatus === 'error');
  el.textContent = !img ? 'Select an image to edit its caption.' :
    sessionSaveState === 'error' ? 'Browser save failed. Export caption files to keep your edits.' :
    sessionSaveState === 'saving' ? 'Saving in this browser…' :
    img.sidecarStatus === 'error' ? 'Caption file save failed. Try Save caption files again.' :
    img.sidecarStatus === 'saved' ? 'Saved in this browser and to .txt.' :
    img.sidecarStatus === 'pending' ? 'Kept in this browser · .txt needs saving.' :
    img.caption ? 'Kept in this browser.' : 'Ready for a caption.';
}

document.getElementById('captionOutput').addEventListener('input', () => {
  const img = state.images[state.activeIdx];
  if (img) {
    img.caption = document.getElementById('captionOutput').value;
    img.editRevision = (img.editRevision || 0) + 1;
    img.sidecarStatus = 'pending';
    if (img.status !== 'running') img.status = img.caption.trim() ? 'done' : 'pending';
    renderQueue();
    scheduleSaveSession();
  }
  updateTokenCount();
  updateControls();
});

document.getElementById('captionOutput').addEventListener('change', async () => {
  if (state.activeIdx === null || !state.saveSidecars || !state.sidecarDirectory) return;
  const img = state.images[state.activeIdx];
  if (!img) return;
  try {
    await writeSidecar(img);
    renderQueue();
    scheduleSaveSession();
  } catch(e) {
    img.sidecarStatus = 'error';
    renderQueue();
    scheduleSaveSession();
    toast(`Could not update ${sidecarFilename(img.name)}: ${e.message}`, 'error-toast');
  }
});

window.addEventListener('pagehide', () => {
  clearTimeout(sessionSaveTimer);
  saveSessionNow();
});

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') {
    clearTimeout(sessionSaveTimer);
    saveSessionNow();
  }
});

async function fileToBase64(file) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result.split(',')[1]);
    r.onerror = rej;
    r.readAsDataURL(file);
  });
}

function captionerTriggerWord() {
  return String(document.getElementById('capTriggerWord').value || '')
    .trim().replace(/\s+/g, ' ').slice(0, 200);
}

const WD_EXCLUDED_TAGS = new Set([
  'masterpiece', 'best_quality', 'high_quality', 'great_quality', 'normal_quality',
  'low_quality', 'worst_quality', 'very_aesthetic', 'aesthetic', 'absurdres',
  'highres', 'incredibly_absurdres', 'tagme', 'translation_request',
  'commentary_request', 'commentary', 'artist_name', 'signature', 'watermark',
  'username', 'dated', 'web_address'
]);

function normalizeWdTag(value) {
  return String(value || '').trim().toLowerCase()
    .replace(/[\s-]+/g, '_').replace(/^_+|_+$/g, '');
}

function readableWdTag(value) {
  return normalizeWdTag(value).replace(/_/g, ' ');
}

function constantWdTraits() {
  return String(document.getElementById('capConstantTraits')?.value || '')
    .split(/[,;\n]+/).map(normalizeWdTag).filter(Boolean);
}

function wdScoreThreshold() {
  const value = Number(document.getElementById('capWdThreshold')?.value || 0.40);
  return Math.max(0.05, Math.min(0.95, Number.isFinite(value) ? value : 0.40));
}

function isExcludedWdTag(name) {
  return WD_EXCLUDED_TAGS.has(name) ||
    /^(score_|rating[:_]|source[:_]|quality[:_])/.test(name) ||
    /_quality$/.test(name);
}

function filterWdTags(tags, threshold, constantTraits) {
  const constants = new Set((constantTraits || []).map(normalizeWdTag));
  const seen = new Set();
  return (Array.isArray(tags) ? tags : [])
    .filter(tag => String(tag?.category || '').toLowerCase() === 'general')
    .map(tag => ({
      name: normalizeWdTag(tag?.name),
      score: Number(tag?.score || 0)
    }))
    .filter(tag => tag.name && tag.score >= threshold &&
      !constants.has(tag.name) && !isExcludedWdTag(tag.name))
    .sort((a, b) => b.score - a.score)
    .filter(tag => {
      if (seen.has(tag.name)) return false;
      seen.add(tag.name);
      return true;
    })
    .slice(0, 120);
}

async function tagImageWithWd(imageB64, signal) {
  if (!state.autoTagger.available || !state.autoTagger.ready || !state.autoTagger.runtimeReady) {
    await loadAutoTaggerStatus();
  }
  updateWdHybridUi();
  if (!state.autoTagger.available) throw new Error('Auto Tagger is not installed');
  if (!state.autoTagger.runtimeReady) {
    throw new Error(state.autoTagger.status?.runtime?.message || 'Auto Tagger runtime is unavailable');
  }
  if (!state.autoTagger.ready) throw new Error('Install the WD model in Auto Tagger first');

  const threshold = wdScoreThreshold();
  const resp = await fetch('/api/auto_tagger/tag-image', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_b64: imageB64, general_threshold: threshold }),
    signal: signal
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok || data.error) {
    if (resp.status === 409 || resp.status === 503) await loadAutoTaggerStatus();
    throw new Error(data.error || `WD tagging failed (${resp.status})`);
  }
  return filterWdTags(data.tags, threshold, constantWdTraits());
}

function renderCaptionerPrompt(prompt, trigger, context={}) {
  let text = String(prompt || '');
  text = text.replace(/\{WD_TAGS\}/gi, () => context.wdTags || '(none above threshold)');
  text = text.replace(/\{CONSTANT_TRAITS\}/gi, () => context.constantTraits || '(none supplied)');
  if (trigger) {
    if (/\{TRIGGER\}/i.test(text)) {
      return text.replace(/\{TRIGGER\}/gi, () => trigger);
    }
    return text.trimEnd() + `\n\nBegin the final caption with this exact trigger token: ${trigger}`;
  }
  return text.split('\n')
    .filter(line => !/\{TRIGGER\}/i.test(line))
    .join('\n').replace(/\n{3,}/g, '\n\n').trim();
}

function ensureCaptionerTrigger(caption, trigger) {
  const text = String(caption || '').trim();
  if (!trigger) return text;
  const escaped = trigger.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const prefixMatch = text.match(new RegExp('^' + escaped + '(?=$|[\\s,.:;!?])'));
  if (prefixMatch) {
    const remainder = text.slice(prefixMatch[0].length).replace(/^[\s,.:;!?-]+/, '');
    return trigger + (remainder ? ', ' + remainder : '');
  }
  return trigger + (text ? ', ' + text : '');
}

function normalizeWdHybridCaption(caption, trigger) {
  const text = String(caption || '').trim();
  if (!text) return text;
  if (!trigger) return text.replace(/_/g, ' ');
  const escaped = trigger.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const hasExactPrefix = new RegExp('^' + escaped + '(?=$|[\\s,.:;!?])').test(text);
  if (!hasExactPrefix) return text.replace(/_/g, ' ');
  return trigger + text.slice(trigger.length).replace(/_/g, ' ');
}

function buildDirectCaptionPayload(imageB64, mediaType, systemPrompt) {
  const cfg = state.config || {};
  const payload = {
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: [
        { type: 'image_url', image_url: { url: `data:${mediaType};base64,${imageB64}` } },
        { type: 'text', text: 'Caption this image according to the system instructions.' }
      ] }
    ]
  };
  if (cfg.model) payload.model = cfg.model;
  ['temperature', 'top_p', 'max_tokens', 'top_k', 'presence_penalty'].forEach(key => {
    if (cfg[key] != null) payload[key] = cfg[key];
  });
  return { cfg, payload };
}

async function captionViaBrowser(imageB64, mediaType, trigger, systemPrompt, signal) {
  const request = buildDirectCaptionPayload(imageB64, mediaType, systemPrompt);
  const controller = new AbortController();
  let timedOut = false;
  const abortFromBatch = () => controller.abort();
  if (signal?.aborted) controller.abort();
  else signal?.addEventListener('abort', abortFromBatch, { once: true });
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, 120000);
  const post = payload => fetch(request.cfg.api_url + '/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal: controller.signal
  });

  try {
    let resp = await post(request.payload);
    if (!resp.ok) {
      const errorText = await resp.text();
      if (/top_k|presence_penalty|unsupported|unknown.*(param|field|key)/i.test(errorText)) {
        const fallback = Object.assign({}, request.payload);
        delete fallback.top_k;
        delete fallback.presence_penalty;
        resp = await post(fallback);
      } else {
        throw new Error(errorText || `LM Studio returned HTTP ${resp.status}`);
      }
    }
    if (!resp.ok) throw new Error((await resp.text()) || `LM Studio returned HTTP ${resp.status}`);
    const data = await resp.json();
    const caption = data?.choices?.[0]?.message?.content;
    if (typeof caption !== 'string') throw new Error('Unexpected LM Studio response');
    return ensureCaptionerTrigger(caption, trigger);
  } catch(e) {
    if (e?.name === 'AbortError' && signal?.aborted) throw e;
    if (e?.name === 'TypeError' || (e?.name === 'AbortError' && timedOut)) e.directUnavailable = true;
    throw e;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', abortFromBatch);
  }
}

async function captionViaHub(imageB64, mediaType, trigger, systemPrompt, signal) {
  const resp = await fetch('/api/captioner/caption', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image_b64: imageB64,
      media_type: mediaType,
      override_prompt: systemPrompt,
      trigger_word: ''
    }),
    signal: signal
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok || data.error) throw new Error(data.error || `Caption request failed (${resp.status})`);
  return ensureCaptionerTrigger(data.caption, trigger);
}

function throwIfCaptioningStopped(signal) {
  if (signal?.aborted) throw new DOMException('Captioning stopped', 'AbortError');
}

async function captionImage(img, signal) {
  const editRevision = img.editRevision || 0;
  img.status = 'running';
  renderQueue();
  scheduleSaveSession();
  try {
    throwIfCaptioningStopped(signal);
    const b64 = await fileToBase64(img.file);
    throwIfCaptioningStopped(signal);
    const mediaType = img.file.type || 'image/jpeg';
    const trigger = captionerTriggerWord();
    const context = {};
    if (isWdHybridPreset()) {
      img.stage = 'WD tagging';
      renderQueue();
      const wdTags = await tagImageWithWd(b64, signal);
      context.wdTags = wdTags.map(tag => readableWdTag(tag.name)).join(', ');
      context.constantTraits = constantWdTraits().map(readableWdTag).join(', ');
    }
    throwIfCaptioningStopped(signal);
    img.stage = 'captioning';
    renderQueue();
    const systemPrompt = renderCaptionerPrompt(state.activeOverride || '', trigger, context);
    let generatedCaption;
    try {
      generatedCaption = await captionViaBrowser(b64, mediaType, trigger, systemPrompt, signal);
    } catch(e) {
      if (!e.directUnavailable) throw e;
      throwIfCaptioningStopped(signal);
      generatedCaption = await captionViaHub(b64, mediaType, trigger, systemPrompt, signal);
    }
    if (isWdHybridPreset()) {
      generatedCaption = normalizeWdHybridCaption(generatedCaption, trigger);
    }
    if ((img.editRevision || 0) === editRevision) {
      img.caption = generatedCaption;
      img.sidecarStatus = '';
    }
    img.status = img.caption.trim() ? 'done' : 'pending';
    img.stage = '';
    if (state.saveSidecars && state.sidecarDirectory) {
      try {
        await writeSidecar(img);
      } catch(sidecarError) {
        img.sidecarStatus = 'error';
        toast(`Caption ready, but ${sidecarFilename(img.name)} could not be saved: ${sidecarError.message}`, 'error-toast');
      }
    }
  } catch (e) {
    const stopped = e?.name === 'AbortError' && signal?.aborted;
    img.status = (img.editRevision || 0) !== editRevision && img.caption.trim() ? 'done' : (stopped ? 'pending' : 'error');
    img.stage = '';
    if (!stopped) {
      console.error(e);
      toast(`Caption failed: ${e.message}`, 'error-toast');
    }
  }
  renderQueue();
  scheduleSaveSession();
  // Manual edits win over a generation started before that edit.
  const ta = document.getElementById('captionOutput');
  if (state.activeIdx === state.images.indexOf(img) && ta.value !== img.caption) {
    ta.value = img.caption;
    updateTokenCount();
  }
}

async function runOne() {
  if (state.activeIdx === null || state.running) return;
  const img = state.images[state.activeIdx];
  state.running = true;
  state.stopRequested = false;
  updateControls();
  try {
    await prepareSidecarDirectory();
    if (state.stopRequested) return;
    const controller = new AbortController();
    state.activeRequestController = controller;
    await captionImage(img, controller.signal);
  } finally {
    state.activeRequestController = null;
    state.running = false;
    state.stopRequested = false;
    updateControls();
  }
}

async function runCaptionBatch(images, completedMessage) {
  if (state.running) return;
  if (!images.length) {
    toast('All images already have captions', 'success');
    return;
  }
  state.running = true;
  state.stopRequested = false;
  updateControls();
  try {
    await prepareSidecarDirectory();
    for (const img of images) {
      if (state.stopRequested) break;
      const controller = new AbortController();
      state.activeRequestController = controller;
      await captionImage(img, controller.signal);
      if (state.activeRequestController === controller) state.activeRequestController = null;
    }
  } finally {
    state.activeRequestController = null;
    state.running = false;
    const wasStopped = state.stopRequested;
    state.stopRequested = false;
    updateControls();
    if (wasStopped) {
      const done = state.images.filter(i => i.status === 'done').length;
      toast(`Stopped · ${done} of ${state.images.length} captioned`);
    } else {
      toast(completedMessage, 'success');
    }
  }
}

async function runAll() {
  return runCaptionBatch(
    state.images.filter(img => img.status !== 'done'),
    'All done!'
  );
}

async function runUncaptioned() {
  return runCaptionBatch(
    state.images.filter(img => !String(img.caption || '').trim()),
    'All uncaptioned images are done!'
  );
}

function stopAll() {
  if (!state.running || state.stopRequested) return;
  state.stopRequested = true;
  updateControls();
  state.activeRequestController?.abort();
}

function clearAll() {
  if (state.running) return;
  state.images.forEach(i => URL.revokeObjectURL(i.url));
  state.images = [];
  state.activeIdx = null;
  state.sidecarDirectory = null;
  state.sidecarDirectoryName = '';
  updateSidecarUi();
  renderQueue();
  document.getElementById('imagePanel').innerHTML = `
    <div class="cap-empty">
      <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
        <rect x="3" y="3" width="18" height="18" rx="2"/>
        <circle cx="8.5" cy="8.5" r="1.5"/>
        <polyline points="21 15 16 10 5 21"/>
      </svg>
      <p>No image selected.<br>Add images from the sidebar.</p>
    </div>`;
  document.getElementById('captionOutput').value = '';
  document.getElementById('currentFile').textContent = 'no file selected';
  document.getElementById('tokenCount').textContent = '— words';
  document.getElementById('runOneBtn').disabled = true;
  updateControls();
  scheduleSaveSession();
}

function copyCaption() {
  const txt = document.getElementById('captionOutput').value.trim();
  if (!txt) return toast('No caption to copy', 'error-toast');
  (function(v){function fb(){try{const e=document.createElement('textarea');e.value=v;e.style.position='fixed';e.style.opacity='0';document.body.appendChild(e);e.select();document.execCommand('copy');document.body.removeChild(e);toast('Copied!','success');}catch(err){toast('Copy failed','error-toast');}}if(navigator.clipboard&&window.isSecureContext){navigator.clipboard.writeText(v).then(()=>toast('Copied!','success')).catch(fb);}else{fb();}})(txt);
}

function copyAll() {
  const done = state.images.filter(i => i.caption);
  if (!done.length) return toast('No captions yet', 'error-toast');
  const txt = done.map(i => i.caption).join('\n\n');
  navigator.clipboard.writeText(txt).then(() => toast(`Copied ${done.length} captions`, 'success'));
}

async function saveCurrentToLibrary() {
  if (state.activeIdx === null) return toast('No image selected', 'error-toast');
  const img = state.images[state.activeIdx];
  if (!img || !img.caption) return toast('No caption yet for this image', 'error-toast');
  await postCardToLibrary(img);
  toast('Saved to library', 'success');
}

async function saveAllToLibrary() {
  const done = state.images.filter(i => i.caption);
  if (!done.length) return toast('No captions to save', 'error-toast');
  if (!confirm(`Save ${done.length} caption(s) to the library?`)) return;
  let ok = 0;
  for (const img of done) {
    try { await postCardToLibrary(img); ok++; } catch(e) {}
  }
  toast(`Saved ${ok} of ${done.length} to library`, ok === done.length ? 'success' : 'error-toast');
}

async function postCardToLibrary(img) {
  /* The VL captioner outputs an image-generation prompt. Save it as 'generation'
     so it lands in the right bucket; reserve 'captioner' for instruction presets. */
  let attach = null;
  if (img.file) {
    try {
      const fd = new FormData();
      fd.append('file', img.file, img.name);
      const up = await fetch('/api/library/attachment', { method: 'POST', body: fd }).then(r => r.json());
      if (up && up.ok) {
        attach = { attachment: up.attachment, attachment_type: up.attachment_type, attachment_name: up.attachment_name };
      }
    } catch (e) { /* ignore — save card without attachment */ }
  }
  const targetSel = document.getElementById('saveTargetSelect');
  const target = (targetSel && targetSel.value) || 'general';
  const body = {
    type: 'generation',
    target: target,
    title: img.name.replace(/\.[^.]+$/, ''),
    content: img.caption,
    source_image: img.name,
    tags: ['caption-output', 'vl-captioner']
  };
  if (attach) Object.assign(body, attach);
  return fetch('/api/library/card', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  }).then(r => r.json());
}

async function saveSidecarsNow() {
  const done = state.images.filter(i => i.caption);
  if (!done.length) return toast('No captions to save', 'error-toast');

  if (!supportsWritableFolders()) return exportSidecarsZip(done);
  try {
    if (!state.sidecarDirectory) await chooseSidecarDirectory();
    if (!state.sidecarDirectory) return;
    let saved = 0;
    for (const img of done) {
      await writeSidecar(img);
      saved++;
    }
    renderQueue();
    scheduleSaveSession();
    toast(`Saved ${saved} caption file${saved === 1 ? '' : 's'} to ${state.sidecarDirectoryName}`, 'success');
  } catch(e) {
    if (e && e.name === 'AbortError') return;
    toast(`Caption file save failed: ${e.message}`, 'error-toast');
  }
}

async function exportSidecarsZip(done) {
  try {
    const response = await fetch('/api/captioner/sidecars', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ items: done.map(img => ({ name: img.name, caption: img.caption })) })
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || `Export failed (${response.status})`);
    }
    downloadBlob('caption-files.zip', await response.blob());
    toast(`Exported ${done.length} caption files`, 'success');
  } catch(e) {
    toast(`Caption file export failed: ${e.message}`, 'error-toast');
  }
}

function exportTxt() {
  const done = state.images.filter(i => i.caption);
  if (!done.length) return toast('No captions to export', 'error-toast');
  const txt = done.map(i => i.caption).join('\n');
  download('captions.txt', txt, 'text/plain');
  toast('Exported captions.txt', 'success');
}

function exportCsv() {
  const done = state.images.filter(i => i.caption);
  if (!done.length) return toast('No captions to export', 'error-toast');
  const rows = [['filename', 'caption'], ...done.map(i => [i.name, i.caption])];
  const csv = rows.map(r => r.map(v => `"${v.replace(/"/g,'""')}"`).join(',')).join('\n');
  download('captions.csv', csv, 'text/csv');
  toast('Exported captions.csv', 'success');
}

function download(filename, content, type) {
  downloadBlob(filename, new Blob([content], { type }));
}

function downloadBlob(filename, blob) {
  const a = document.createElement('a');
  const url = URL.createObjectURL(blob);
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

let toastTimer;
function toast(msg, type='') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'cap-toast show' + (type ? ' ' + type : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.className = 'cap-toast', 2500);
}

init();
</script>
"""
