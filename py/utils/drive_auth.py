from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow


SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
]

CLIENT_SECRET = Path("credentials/client_secret.json")
TOKEN_PATH = Path("credentials/drive_token.json")


def main():
    if not CLIENT_SECRET.exists():
        raise FileNotFoundError(f"OAuth client tidak ditemukan: {CLIENT_SECRET}")

    flow = InstalledAppFlow.from_client_secrets_file(
        CLIENT_SECRET,
        SCOPES,
    )

    credentials = flow.run_local_server(
        port=0,
        access_type="offline",
        prompt="consent",
    )

    TOKEN_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    TOKEN_PATH.write_text(credentials.to_json())

    print()
    print("[✓] Google Drive OAuth berhasil.")
    print(f"[✓] Token: {TOKEN_PATH}")


if __name__ == "__main__":
    main()
