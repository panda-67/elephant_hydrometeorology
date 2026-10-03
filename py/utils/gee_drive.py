from __future__ import annotations

import time
import zipfile
from pathlib import Path

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


DRIVE_SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
]

DEFAULT_FOLDER = "GeoForensic_Tangse_Meureudu"


def get_drive_service(
    token_path: str | Path = "credentials/drive_token.json",
):
    """Create Google Drive API service."""

    token_path = Path(token_path)

    if not token_path.exists():
        raise FileNotFoundError(f"Google Drive token tidak ditemukan: {token_path}")

    credentials = Credentials.from_authorized_user_file(
        token_path,
        DRIVE_SCOPES,
    )

    return build(
        "drive",
        "v3",
        credentials=credentials,
        cache_discovery=False,
    )


def get_folder_id(
    service,
    folder_name: str = DEFAULT_FOLDER,
):
    """Find Google Drive folder by name."""

    query = (
        "trashed = false "
        "and mimeType = 'application/vnd.google-apps.folder' "
        f"and name = '{folder_name}'"
    )

    response = (
        service.files()
        .list(
            q=query,
            spaces="drive",
            fields="files(id,name)",
            pageSize=100,
        )
        .execute()
    )

    folders = response.get("files", [])

    if not folders:
        raise FileNotFoundError(f"Folder Google Drive tidak ditemukan: {folder_name}")

    return folders[0]["id"]


def wait_for_earth_engine_task(
    task,
    poll_interval: int = 15,
):
    """
    Menunggu Earth Engine Export Task sampai selesai.

    Returns
    -------
    str
        Status akhir task.
    """

    print("    [~] Waiting for Earth Engine task...")

    while True:
        status = task.status()

        state = status.get("state", "UNKNOWN")

        print(
            f"\r    [~] Earth Engine task: {state}",
            end="",
            flush=True,
        )

        if state == "COMPLETED":
            print()
            print("    [✓] Earth Engine export completed.")
            return state

        if state in {
            "FAILED",
            "CANCELLED",
        }:
            print()

            error_message = status.get(
                "error_message",
                "Unknown Earth Engine error.",
            )

            raise RuntimeError(f"Earth Engine task {state}: {error_message}")

        time.sleep(poll_interval)


def find_existing_drive_file(
    service,
    filename_prefix: str,
    folder_id: str,
):
    """
    Mencari file yang sudah ada di folder Drive.

    Tidak menunggu. Mengembalikan None jika tidak ditemukan.
    """

    escaped_prefix = filename_prefix.replace(
        "'",
        "\\'",
    )

    query = (
        f"'{folder_id}' in parents "
        "and trashed = false "
        f"and name contains '{escaped_prefix}'"
    )

    response = (
        service.files()
        .list(
            q=query,
            spaces="drive",
            fields=("files(id,name,size,mimeType,createdTime,modifiedTime)"),
            pageSize=100,
        )
        .execute()
    )

    files = response.get("files", [])

    matching = [file for file in files if file["name"].startswith(filename_prefix)]

    if not matching:
        return None

    matching.sort(
        key=lambda x: x.get(
            "modifiedTime",
            "",
        ),
        reverse=True,
    )

    return matching[0]


def find_drive_file(
    service,
    filename_prefix: str,
    folder_id: str,
    timeout: int = 600,
    poll_interval: int = 10,
):
    """
    Wait for an Earth Engine exported file
    to appear in the target Drive folder.
    """

    escaped_prefix = filename_prefix.replace(
        "'",
        "\\'",
    )

    query = (
        f"'{folder_id}' in parents "
        "and trashed = false "
        f"and name contains '{escaped_prefix}'"
    )

    start = time.time()

    while time.time() - start < timeout:
        response = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields=("files(id,name,size,mimeType,createdTime,modifiedTime)"),
                pageSize=100,
            )
            .execute()
        )

        files = response.get("files", [])

        matching = [file for file in files if file["name"].startswith(filename_prefix)]

        if matching:
            matching.sort(
                key=lambda x: x.get(
                    "modifiedTime",
                    "",
                ),
                reverse=True,
            )

            selected = matching[0]

            print(f"    [✓] Drive file found: {selected['name']}")

            return selected

        print(
            f"\r    [~] Waiting for Drive file '{filename_prefix}'...",
            end="",
            flush=True,
        )

        time.sleep(poll_interval)

    print()

    raise TimeoutError(
        f"File '{filename_prefix}' tidak ditemukan "
        f"di folder Drive dalam {timeout} detik."
    )


def download_existing_drive_export(
    filename_prefix: str,
    output_path: str | Path,
    folder_name: str = DEFAULT_FOLDER,
    token_path: str | Path = "credentials/drive_token.json",
):
    """
    Mencari dan mengunduh file export yang sudah ada
    di Google Drive.

    Returns
    -------
    str | None
        Path lokal jika file ditemukan,
        None jika tidak ada.
    """

    service = get_drive_service(token_path)

    folder_id = get_folder_id(
        service,
        folder_name,
    )

    drive_file = find_existing_drive_file(
        service=service,
        filename_prefix=filename_prefix,
        folder_id=folder_id,
    )

    if drive_file is None:
        return None

    print(f"    [✓] Existing Drive file found: {drive_file['name']}")

    output_path = Path(output_path)

    temporary_path = output_path.parent / f".{drive_file['name']}"

    download_drive_file(
        service=service,
        file_id=drive_file["id"],
        output_path=temporary_path,
    )

    if zipfile.is_zipfile(temporary_path):
        print("    [~] Extracting GeoTIFF...")

        with zipfile.ZipFile(
            temporary_path,
            "r",
        ) as archive:
            tif_files = [
                name
                for name in archive.namelist()
                if name.lower().endswith((".tif", ".tiff"))
            ]

            if not tif_files:
                raise RuntimeError("ZIP Drive tidak mengandung GeoTIFF.")

            with (
                archive.open(tif_files[0]) as src,
                open(output_path, "wb") as dst,
            ):
                while True:
                    chunk = src.read(1024 * 1024)

                    if not chunk:
                        break

                    dst.write(chunk)

        temporary_path.unlink(missing_ok=True)

    else:
        temporary_path.replace(output_path)

    print(f"    [✓] Existing export ready: {output_path}")

    return str(output_path)


def download_drive_file(
    service,
    file_id: str,
    output_path: str | Path,
):
    """Download a Google Drive file."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    request = service.files().get_media(
        fileId=file_id,
    )

    with open(
        output_path,
        "wb",
    ) as destination:
        downloader = MediaIoBaseDownload(
            destination,
            request,
            chunksize=10 * 1024 * 1024,
        )

        done = False

        while not done:
            status, done = downloader.next_chunk()

            if status:
                progress = int(status.progress() * 100)

                print(
                    f"\r    [~] Downloading from Drive: {progress:3d}%",
                    end="",
                    flush=True,
                )

    print()

    print(f"    [✓] Downloaded: {output_path}")

    return str(output_path)


def download_earth_engine_export(
    task,
    filename_prefix: str,
    output_path: str | Path,
    folder_name: str = DEFAULT_FOLDER,
    task_timeout: int = 7200,
    drive_timeout: int = 600,
    poll_interval: int = 15,
    token_path: str | Path = "credentials/drive_token.json",
):
    """
    Complete Earth Engine → Google Drive → local workflow.

    1. Wait for Earth Engine task.
    2. Connect to Google Drive.
    3. Find exported file.
    4. Download file locally.
    5. Extract GeoTIFF if Drive contains ZIP.
    """

    output_path = Path(output_path)

    # ------------------------------------------------------------
    # 1. Wait for Earth Engine
    # ------------------------------------------------------------

    wait_start = time.time()

    wait_for_earth_engine_task(
        task,
        poll_interval=poll_interval,
    )

    if time.time() - wait_start > task_timeout:
        raise TimeoutError("Earth Engine task melebihi batas waktu.")

    # ------------------------------------------------------------
    # 2. Google Drive
    # ------------------------------------------------------------

    print("    [~] Connecting to Google Drive...")

    service = get_drive_service(token_path)

    folder_id = get_folder_id(
        service,
        folder_name,
    )

    # ------------------------------------------------------------
    # 3. Find exported file
    # ------------------------------------------------------------

    drive_file = find_drive_file(
        service=service,
        filename_prefix=filename_prefix,
        folder_id=folder_id,
        timeout=drive_timeout,
        poll_interval=10,
    )

    # ------------------------------------------------------------
    # 4. Download
    # ------------------------------------------------------------

    temporary_path = output_path.parent / f".{drive_file['name']}"

    download_drive_file(
        service=service,
        file_id=drive_file["id"],
        output_path=temporary_path,
    )

    # ------------------------------------------------------------
    # 5. Extract if ZIP
    # ------------------------------------------------------------

    if zipfile.is_zipfile(temporary_path):
        print("    [~] Extracting GeoTIFF...")

        with zipfile.ZipFile(
            temporary_path,
            "r",
        ) as archive:
            tif_files = [
                name
                for name in archive.namelist()
                if name.lower().endswith((".tif", ".tiff"))
            ]

            if not tif_files:
                raise RuntimeError(
                    "File ZIP dari Google Drive tidak mengandung GeoTIFF."
                )

            source_tif = tif_files[0]

            with (
                archive.open(source_tif) as src,
                open(output_path, "wb") as dst,
            ):
                while True:
                    chunk = src.read(1024 * 1024)

                    if not chunk:
                        break

                    dst.write(chunk)

        temporary_path.unlink(missing_ok=True)

    else:
        temporary_path.replace(output_path)

    print(f"    [✓] GeoTIFF ready: {output_path}")

    return str(output_path)
