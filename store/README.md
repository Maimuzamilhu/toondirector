# Microsoft Store edition

Same code as the full app, with one difference: it contains no YouTube downloader. A pasted link only reads the video's
public title, and you can paste a transcript by hand. Cartoon from a video takes files only.

## Before the final build

1. In `app/config.py`, set `SUPPORT_EMAIL` and `PRIVACY_URL`.
2. In `store/privacy-policy.html`, replace `EFFECTIVE_DATE` and `SUPPORT_EMAIL`, then publish the page.
3. In `store/listing.md`, replace `TEST_KEY_HERE` with a separate key that has a small balance.

## Build

```
set STORE_EDITION=1
.venv\Scripts\pyinstaller packaging\toondirector.spec --noconfirm --distpath dist_store --workpath build_store
set STORE_EDITION=
.venv\Scripts\python packaging\build_store.py --name "..." --publisher "CN=..." --display "..." --makeappx tools\buildtools\bin\10.0.28000.0\x64\makeappx.exe
```

The three values are on the app's Product identity page in Partner Center.
