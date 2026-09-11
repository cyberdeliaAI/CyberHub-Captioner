# CyberHub Captioner

Create local image captions with an OpenAI-compatible vision model.

- Version: `1.6.1`
- Channel: `stable`
- Publisher: `official`

## Installation

1. Open **Module Manager** in CyberHub.
2. Click **Check for updates**.
3. Find **Captioner** and choose **Install** or **Update**.
4. Restart CyberHub when the installation finishes.

The ZIP attached to this repository's GitHub Release can also be imported manually through Settings.

## Python Packages

- `requests>=2.28`

## Privacy

CyberHub runs locally. A module uses an external service only when its function requires it and the user starts that action.

## Captioner 1.6.1

Adds **Download images + captions**, a ZIP export of original captioned images
and matching caption files, including manual edits. The export runs locally
in the browser and keeps image/caption pairs together when names overlap.

Captioner also includes a connection settings dialog, a dedicated image queue,
editable captions with session recovery, system-prompt JSON sharing, and saved
preset management with duplicate-name protection.

### Building a package

Build an installable ZIP directly from this repository:

```sh
python3 tools/build_test_package.py --output /path/to/packages
```

Import `cyberhub_module_captioner_v1.6.1.zip` through **Settings**, then restart
CyberHub. The ZIP contains only Captioner runtime files and its package
manifest; it preserves user settings and the Library.

### Workspace and connection settings

- Open **Settings** at the top of Captioner, or click the **Connection** status
  in the left column. Both open the same settings dialog.
- Set the API URL and optional model. URLs with or without `/v1` are accepted.
  Empty model and generation fields use the server defaults.
- **Detect model** fills in the first available model. **Save settings** applies
  the form. **Cancel**, Escape, or clicking outside the dialog discards edits.
  Clearing generation overrides also requires **Save settings**.
- Detection tries the browser first, then CyberHub for the saved API URL. When
  entering a different URL that the browser cannot reach directly, save it first
  and retry detection through CyberHub.
- Input and Vision scripts occupy the first column. The adjacent narrow queue
  has its own scrollbar, image count, progress and batch controls. The image
  preview and caption editor fill the remaining space.
- At smaller widths, preview and editor stack; on mobile, the input, queue and
  workspace stack vertically. Both dark and light Hub themes are supported.

### Managing saved presets

- **Save as preset** creates a new Library preset. Its name must be unique among
  saved Captioner presets. The server compares names without case differences,
  repeated whitespace or Unicode compatibility differences. The check includes
  legacy tagged presets and the complete Library, beyond the first 200 entries.
- Use **Update** to change the selected saved preset. Renaming it to another
  preset's name is rejected; its original prompt is kept when saving fails.
- **Delete preset** appears for saved presets. After confirmation, it removes
  that specific preset from the Library and resets the editor to the default
  built-in script. It does not remove images or generated captions. Built-in
  scripts and imported drafts cannot be deleted as Library presets.
- Existing duplicates are not merged or removed automatically. They show their
  Library ID in the dropdown so you can inspect and delete the intended item.
  Existing duplicates may still be edited without renaming them.
- Captioner serializes preset writes across tabs and prevents repeated clicks
  while a write is pending. This is a Captioner rule; the general-purpose Library
  module's own name policy and other modules are unchanged.

### Sharing system prompts

The Vision script section now has a **Prompt title** field and **Import JSON** /
**Export JSON** actions. Export includes the current title and edited prompt.
Import loads a draft in the editor, without running it or replacing a saved
Library preset. **Save as preset** uses the title field to create a new preset;
**Update** also applies title changes to an existing preset.

Files identify themselves as CyberHub `system_prompt` documents with `schema: 1`
and `module: "captioner"`. Captioner rejects other module types, including future
Prompt Engineer files. Imported prompts work without Prompt Library and remain
in the local browser session even before images are loaded.

See [the JSON format](docs/prompt-json-format.md) and
[the example prompt](examples/natural-dataset-caption.captioner.json).

### Editing captions and saving files

Hand edits immediately update the caption for the selected image. The browser
saves the batch, images and edited captions in IndexedDB after a short debounce,
so edits are restored when returning to Captioner or reloading in the same
browser and origin. The editor reports browser-storage failures; export files
if storage is unavailable. Clearing browser data also clears the saved session.

**Save caption files** uses the latest edited captions, including manually
written captions. With writable folder access it writes one matching `.txt`
file per image; otherwise it downloads `caption-files.zip`. Explicit caption
exports skip empty captions. Text is trimmed and files end with a newline.
Copy, Library saves and bulk text/CSV exports use the edited captions too.

**Download images + captions** always downloads `images-and-captions.zip`.
It includes only images with nonempty captions and pairs each original image
with a matching `.txt` file containing the caption at the start of the export.
Image bytes, format, resolution and embedded metadata are preserved. Duplicate
names (including the same stem with different image extensions) receive a
shared numeric suffix for the image and its caption. Unsafe filename characters
are replaced; no folder paths are included.

The ZIP is built locally in the browser without uploading images. It supports
up to 10,000 captioned images and a 2 GB ZIP; larger datasets must be exported
in smaller batches. The button shows progress while preparing the download.
If an original image is missing or cannot be read, the export reports an error
instead of downloading an incomplete dataset. A dataset download does not mark
caption files as saved beside the source images.

When **Save .txt beside images** is enabled and folder access has been granted,
leaving the caption editor also updates that image's `.txt` file. The queue and
editor distinguish captions that need a file update from saved files. Folder
access must be granted again after a reload. A ZIP download does not confirm
that files were extracted beside the images.

A generation response cannot overwrite a manual edit made after that request
started. Concurrent automatic and manual file writes are serialized per image,
so an older write cannot finish after a newer one and replace it.

### Validation

Run the module's regression tests from this repository (Python 3 and Node.js
20 or newer; no extra test dependencies):

```sh
python3 -B -m unittest discover -s tests
node --test tests/captioner_session.test.cjs
```

The tests cover edited caption exports, image switching, session records,
storage failures, generation/edit races, file-write races, settings drafts,
JSON round-trips, module/schema validation, prompt-only session restore, preset
name conflicts, simultaneous saves, complete-list checks and scoped deletion.
Dataset ZIP tests use Python's independent ZIP reader to verify original image
bytes, edited captions, CRCs, Unicode names, duplicate-name pairing, export
snapshots, missing images and size limits.
Browser checks additionally cover the two dialog triggers, save/cancel,
model detection, actual IndexedDB restore and responsive dark/light layouts.
Model responses in the local preview were simulated; no live model is required
for these checks.

## License

See `LICENSE.md` and `THIRD-PARTY-NOTICES.md`.
