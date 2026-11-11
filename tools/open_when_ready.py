"""Open the board in the browser as soon as the server answers (for launchers that
can't easily wait themselves, e.g. the Windows .bat). Gives up quietly after 60 s.

    python tools/open_when_ready.py http://127.0.0.1:8000
"""
import sys
import time
import urllib.request
import webbrowser


def main() -> None:
    url = sys.argv[1].rstrip("/")
    for _ in range(120):
        try:
            urllib.request.urlopen(url + "/api/state", timeout=1)
            webbrowser.open(url)
            return
        except Exception:
            time.sleep(0.5)


if __name__ == "__main__":
    main()
