#!/usr/bin/env python3
"""Multi-Platform Automated Video Uploader (YouTube, Facebook, Instagram).

Automatically uploads rendered videos to:
  - YouTube (Data API v3): Resumable chunked video upload, automated SEO title,
    description, chapters, tags, category, privacy, thumbnail & SRT captions.
  - Facebook (Meta Graph API v19.0+): Facebook Page Videos & Reels API,
    resumable binary upload, title, description, hashtags.
  - Instagram (Meta Graph API / Content Publishing API): Reels container creation,
    resumable binary video upload, status polling, and publishing.

Supports:
  - Interactive OAuth authorization helper for YouTube
  - Credential verification & connectivity checks (--check)
  - Dry-run simulation mode (--dry-run)
  - Seamless integration with metadata.json and script.json
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import socketserver
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]


# --------------------------------------------------------------------------
# Environment & Configuration Loader
# --------------------------------------------------------------------------

def load_dotenv_if_present() -> None:
    """Load key-value pairs from .env if present into os.environ."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass

load_dotenv_if_present()


# --------------------------------------------------------------------------
# Platform Uploaders: YouTube
# --------------------------------------------------------------------------

class YouTubeUploader:
    """YouTube Data API v3 Uploader with OAuth2 & Resumable Upload support."""

    TOKEN_URL = "https://oauth2.googleapis.com/token"
    AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status"
    API_BASE = "https://www.googleapis.com/youtube/v3"

    SCOPES = [
        "https://www.googleapis.com/auth/youtube.upload",
        "https://www.googleapis.com/auth/youtube.force-ssl",
    ]

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        refresh_token: str | None = None,
        token_path: str | Path | None = None,
    ) -> None:
        self.client_id = client_id or os.environ.get("YOUTUBE_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("YOUTUBE_CLIENT_SECRET")
        self.refresh_token = refresh_token or os.environ.get("YOUTUBE_REFRESH_TOKEN")
        self.token_path = Path(token_path or os.environ.get("YOUTUBE_TOKEN_PATH") or (ROOT / "youtube_token.json"))
        self.access_token: str | None = os.environ.get("YOUTUBE_ACCESS_TOKEN")

        # Load token file if exists
        if self.token_path.exists():
            try:
                data = json.loads(self.token_path.read_text(encoding="utf-8"))
                self.refresh_token = data.get("refresh_token") or self.refresh_token
                self.access_token = data.get("access_token") or self.access_token
                self.client_id = data.get("client_id") or self.client_id
                self.client_secret = data.get("client_secret") or self.client_secret
            except Exception:
                pass

    def is_configured(self) -> bool:
        """Return True if sufficient credentials exist to attempt YouTube operations."""
        return bool(self.access_token or (self.client_id and self.client_secret and self.refresh_token))

    def refresh_access_token(self) -> str | None:
        """Fetch a fresh access token using the OAuth2 refresh token."""
        if not (self.client_id and self.client_secret and self.refresh_token):
            return self.access_token

        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self.refresh_token,
            "grant_type": "refresh_token",
        }
        try:
            resp = requests.post(self.TOKEN_URL, data=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                self.access_token = data.get("access_token")
                # Save updated access token
                token_data = {
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "refresh_token": self.refresh_token,
                    "access_token": self.access_token,
                    "updated_at": time.time(),
                }
                self.token_path.write_text(json.dumps(token_data, indent=2), encoding="utf-8")
                return self.access_token
            else:
                print(f"  [YouTube] Token refresh failed ({resp.status_code}): {resp.text}", file=sys.stderr)
        except Exception as e:
            print(f"  [YouTube] Token refresh error: {e}", file=sys.stderr)
        return None

    def get_valid_token(self) -> str | None:
        """Return a valid access token, refreshing if necessary."""
        if self.refresh_token and self.client_id and self.client_secret:
            return self.refresh_access_token()
        return self.access_token

    def start_oauth_flow(self, port: int = 8080) -> bool:
        """Interactive browser OAuth authorization flow to generate a refresh token."""
        if not (self.client_id and self.client_secret):
            print("  [YouTube] Error: YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET are required for OAuth setup.", file=sys.stderr)
            return False

        redirect_uri = f"http://localhost:{port}/callback"
        params = {
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(self.SCOPES),
            "access_type": "offline",
            "prompt": "consent",
        }
        auth_url = f"{self.AUTH_URL}?{urllib.parse.urlencode(params)}"
        print("\n" + "=" * 74)
        print("YOUTUBE OAUTH 2.0 AUTHORIZATION SETUP")
        print("=" * 74)
        print("Please open the following link in your web browser to authorize:")
        print(f"\n{auth_url}\n")

        auth_code = None

        class OAuthCallbackHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                nonlocal auth_code
                query = urllib.parse.urlparse(self.path).query
                params = urllib.parse.parse_qs(query)
                if "code" in params:
                    auth_code = params["code"][0]
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(b"<h1>Authorization Successful!</h1><p>You can close this tab and return to your terminal.</p>")
                else:
                    self.send_response(400)
                    self.end_headers()
            def log_message(self, format, *args):
                pass  # Suppress HTTP server logs

        try:
            with socketserver.TCPServer(("localhost", port), OAuthCallbackHandler) as httpd:
                print(f"Waiting for authorization on {redirect_uri}...")
                httpd.handle_request()
        except Exception as e:
            print(f"  [YouTube] Local server failed on port {port}: {e}", file=sys.stderr)
            return False

        if not auth_code:
            print("  [YouTube] Authorization failed: No code received.", file=sys.stderr)
            return False

        # Exchange auth code for tokens
        token_payload = {
            "code": auth_code,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }
        resp = requests.post(self.TOKEN_URL, data=token_payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            self.refresh_token = data.get("refresh_token")
            self.access_token = data.get("access_token")
            token_data = {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token,
                "access_token": self.access_token,
                "updated_at": time.time(),
            }
            self.token_path.write_text(json.dumps(token_data, indent=2), encoding="utf-8")
            print(f"\n✓ YouTube Authorization successful! Tokens saved to {self.token_path.name}")
            return True
        else:
            print(f"  [YouTube] Token exchange failed ({resp.status_code}): {resp.text}", file=sys.stderr)
            return False

    def check_connection(self) -> dict[str, Any]:
        """Test authentication and fetch the authenticated channel information."""
        token = self.get_valid_token()
        if not token:
            return {"configured": False, "status": "missing_credentials", "message": "No YouTube credentials or tokens found"}

        headers = {"Authorization": f"Bearer {token}"}
        try:
            resp = requests.get(
                f"{self.API_BASE}/channels?part=snippet,statistics&mine=true",
                headers=headers,
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("items", [])
                if items:
                    chan = items[0]
                    title = chan["snippet"]["title"]
                    subs = chan.get("statistics", {}).get("subscriberCount", "N/A")
                    return {
                        "configured": True,
                        "status": "connected",
                        "channel_id": chan["id"],
                        "channel_name": title,
                        "subscribers": subs,
                        "message": f"Connected as '{title}' ({subs} subscribers)",
                    }
                return {"configured": True, "status": "no_channel", "message": "Authenticated, but no YouTube channel found for this account"}
            return {"configured": True, "status": "error", "code": resp.status_code, "message": resp.text[:200]}
        except Exception as e:
            return {"configured": True, "status": "network_error", "message": str(e)}

    def upload_video(
        self,
        video_path: str | Path,
        yt_metadata: dict | None = None,
        privacy_status: str = "public",
        publish_at: str | None = None,
        srt_path: str | Path | None = None,
        thumbnail_path: str | Path | None = None,
        dry_run: bool = False,
        meta: dict | None = None,
        privacy: str | None = None,
    ) -> dict[str, Any]:
        """Upload a video file to YouTube with metadata using resumable upload protocol."""
        video_p = Path(video_path).resolve()
        if not dry_run and not video_p.exists():
            return {"success": False, "error": f"Video file not found: {video_p}"}

        effective_privacy = privacy or privacy_status or "public"
        yt_meta = yt_metadata or meta or {}
        file_size = video_p.stat().st_size if video_p.exists() else 0
        title = yt_meta.get("title", "Untitled Video")
        description = yt_meta.get("description", "")
        tags = yt_meta.get("tags", [])
        category_id = str(yt_meta.get("category_id", "24"))
        is_shorts = bool(yt_meta.get("is_shorts", False))

        # Build YouTube API Resource Body
        status_body: dict[str, Any] = {
            "privacyStatus": effective_privacy,
            "selfDeclaredMadeForKids": False,
            "madeForKids": False,
            "embeddable": True,
        }
        if publish_at and effective_privacy == "private":
            status_body["publishAt"] = publish_at

        # YouTube tag requirements:
        # - Must not contain '<' or '>'
        # - No empty strings
        # - Total combined characters across all tags must not exceed ~400-500 chars
        # - Individual tag length < 100 chars
        clean_tags = []
        total_len = 0
        for t in tags:
            tag_s = re.sub(r"[<>\r\n]+", "", str(t)).strip()
            if not tag_s:
                continue
            if len(tag_s) > 80:
                tag_s = tag_s[:80].strip()
            if total_len + len(tag_s) + 1 > 450:
                break
            clean_tags.append(tag_s)
            total_len += len(tag_s) + 1

        snippet_body: dict[str, Any] = {
            "title": title[:100],
            "description": description[:5000],
            "categoryId": category_id,
            "defaultLanguage": yt_meta.get("default_language", "hi"),
            "defaultAudioLanguage": yt_meta.get("default_audio_language", "hi"),
        }
        if clean_tags:
            snippet_body["tags"] = clean_tags

        resource_body = {
            "snippet": snippet_body,
            "status": status_body,
        }

        if dry_run:
            print(f"  [YouTube] [DRY RUN] Would upload {video_p.name} ({file_size / (1024*1024):.2f} MB)")
            print(f"            Title: {title}")
            print(f"            Category: {category_id} | Privacy: {effective_privacy} | Shorts: {is_shorts}")
            print(f"            Tags: {', '.join(tags[:6])}...")
            return {
                "success": True,
                "platform": "youtube",
                "id": "DRY_RUN_YT_ID",
                "video_id": "DRY_RUN_YT_ID",
                "url": "https://youtu.be/DRY_RUN_YT_ID",
                "status": "dry_run_simulated",
            }

        token = self.get_valid_token()
        if not token:
            return {"success": False, "error": "YouTube authentication failed. Run with --auth youtube or set credentials in .env"}

        # Step 1: Initialize Resumable Upload
        init_headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Length": str(file_size),
            "X-Upload-Content-Type": "video/mp4",
        }

        try:
            print(f"  [YouTube] Initiating resumable upload for '{title}'...")
            init_resp = requests.post(
                self.UPLOAD_URL,
                headers=init_headers,
                data=json.dumps(resource_body),
                timeout=45,
            )
            if init_resp.status_code != 200:
                return {
                    "success": False,
                    "error": f"Upload initiation failed ({init_resp.status_code}): {init_resp.text}",
                }

            upload_location = init_resp.headers.get("Location")
            if not upload_location:
                return {"success": False, "error": "No upload location header returned by YouTube API"}

            # Step 2: Stream / Upload video binary
            print(f"  [YouTube] Uploading video binary ({file_size / (1024*1024):.2f} MB)...")
            with open(video_p, "rb") as vf:
                upload_headers = {
                    "Content-Length": str(file_size),
                    "Content-Type": "video/mp4",
                }
                upload_resp = requests.put(
                    upload_location,
                    headers=upload_headers,
                    data=vf,
                    timeout=600,
                )

            if upload_resp.status_code not in (200, 201):
                return {
                    "success": False,
                    "error": f"Video upload binary transfer failed ({upload_resp.status_code}): {upload_resp.text}",
                }

            res_data = upload_resp.json()
            video_id = res_data.get("id")
            video_url = f"https://youtu.be/{video_id}"
            print(f"  [YouTube] ✓ Video uploaded successfully! Video ID: {video_id}")
            print(f"            Watch URL: {video_url}")

            # Step 3: Optional Caption (.srt) Upload
            if srt_path and Path(srt_path).exists() and video_id:
                self.upload_captions(video_id, srt_path, language=yt_metadata.get("default_language", "hi"))

            # Step 4: Optional Thumbnail Upload
            if thumbnail_path and Path(thumbnail_path).exists() and video_id:
                self.upload_thumbnail(video_id, thumbnail_path)

            return {
                "success": True,
                "platform": "youtube",
                "video_id": video_id,
                "url": video_url,
                "privacy": privacy_status,
                "title": title,
            }

        except Exception as e:
            return {"success": False, "error": f"YouTube upload exception: {e}"}

    def upload_captions(self, video_id: str, srt_path: str | Path, language: str = "hi") -> bool:
        """Upload .srt subtitles track to the uploaded YouTube video."""
        srt_p = Path(srt_path).resolve()
        if not srt_p.exists():
            return False
        token = self.get_valid_token()
        if not token:
            return False

        try:
            print(f"  [YouTube] Uploading synchronized subtitles ({srt_p.name})...")
            meta = {
                "snippet": {
                    "videoId": video_id,
                    "language": language,
                    "name": "Hinglish / Hindi Captions",
                    "isDraft": False,
                }
            }
            init_url = f"https://www.googleapis.com/upload/youtube/v3/captions?uploadType=resumable&part=snippet"
            srt_size = srt_p.stat().st_size
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Length": str(srt_size),
                "X-Upload-Content-Type": "*/*",
            }
            r = requests.post(init_url, headers=headers, data=json.dumps(meta), timeout=30)
            if r.status_code == 200 and "Location" in r.headers:
                loc = r.headers["Location"]
                with open(srt_p, "rb") as sf:
                    up_r = requests.put(loc, headers={"Content-Length": str(srt_size), "Content-Type": "*/*"}, data=sf, timeout=60)
                if up_r.status_code in (200, 201):
                    print("  [YouTube] ✓ Subtitles attached successfully!")
                    return True
        except Exception as e:
            print(f"  [YouTube] Subtitle upload note: {e}", file=sys.stderr)
        return False

    def upload_thumbnail(self, video_id: str, image_path: str | Path) -> bool:
        """Set a custom thumbnail image for the uploaded YouTube video."""
        img_p = Path(image_path).resolve()
        if not img_p.exists():
            return False
        token = self.get_valid_token()
        if not token:
            return False
        try:
            print(f"  [YouTube] Setting custom thumbnail ({img_p.name})...")
            url = f"https://www.googleapis.com/upload/youtube/v3/thumbnails/set?videoId={video_id}"
            with open(img_p, "rb") as f:
                r = requests.post(url, headers={"Authorization": f"Bearer {token}", "Content-Type": "image/jpeg"}, data=f, timeout=60)
            if r.status_code in (200, 201):
                print("  [YouTube] ✓ Custom thumbnail set successfully!")
                return True
        except Exception as e:
            print(f"  [YouTube] Thumbnail upload note: {e}", file=sys.stderr)
        return False


# --------------------------------------------------------------------------
# Platform Uploaders: Facebook
# --------------------------------------------------------------------------

class FacebookUploader:
    """Meta Facebook Graph API Uploader for Facebook Page Videos & Reels."""

    GRAPH_BASE = "https://graph.facebook.com/v19.0"

    def __init__(
        self,
        page_id: str | None = None,
        access_token: str | None = None,
    ) -> None:
        self.page_id = page_id or os.environ.get("FACEBOOK_PAGE_ID") or os.environ.get("FB_PAGE_ID")
        self.access_token = access_token or os.environ.get("FACEBOOK_PAGE_ACCESS_TOKEN") or os.environ.get("FB_ACCESS_TOKEN")

    def is_configured(self) -> bool:
        """Return True if Page ID and Page Access Token are present."""
        return bool(self.page_id and self.access_token)

    def check_connection(self) -> dict[str, Any]:
        """Verify Facebook Page Access Token and retrieve page details."""
        if not self.is_configured():
            return {"configured": False, "status": "missing_credentials", "message": "No FACEBOOK_PAGE_ID or FACEBOOK_PAGE_ACCESS_TOKEN found in .env"}

        url = f"{self.GRAPH_BASE}/{self.page_id}?fields=id,name,fan_count,link&access_token={self.access_token}"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                name = data.get("name", "Facebook Page")
                followers = data.get("fan_count", "N/A")
                return {
                    "configured": True,
                    "status": "connected",
                    "page_id": data.get("id"),
                    "page_name": name,
                    "followers": followers,
                    "message": f"Connected to Facebook Page '{name}' (ID: {self.page_id})",
                }
            return {"configured": True, "status": "error", "code": resp.status_code, "message": resp.text[:200]}
        except Exception as e:
            return {"configured": True, "status": "network_error", "message": str(e)}

    def upload_video(
        self,
        video_path: str | Path,
        fb_metadata: dict | None = None,
        is_reel: bool = True,
        dry_run: bool = False,
        meta: dict | None = None,
    ) -> dict[str, Any]:
        """Upload video/reel to Facebook Page using Meta Graph API."""
        video_p = Path(video_path).resolve()
        if not dry_run and not video_p.exists():
            return {"success": False, "error": f"Video file not found: {video_p}"}

        fb_meta = fb_metadata or meta or {}
        file_size = video_p.stat().st_size if video_p.exists() else 0
        title = fb_meta.get("title", "Untitled Video")
        description = fb_meta.get("description", "")

        if dry_run:
            print(f"  [Facebook] [DRY RUN] Would upload {video_p.name} ({file_size / (1024*1024):.2f} MB)")
            print(f"             Page ID: {self.page_id} | Type: {'Reel' if is_reel else 'Video'}")
            print(f"             Title: {title}")
            return {
                "success": True,
                "platform": "facebook",
                "id": "DRY_RUN_FB_ID",
                "video_id": "DRY_RUN_FB_ID",
                "url": f"https://www.facebook.com/{self.page_id or 'page'}/videos/DRY_RUN_FB_ID",
                "status": "dry_run_simulated",
            }

        if not self.is_configured():
            return {"success": False, "error": "Facebook credentials missing. Add FACEBOOK_PAGE_ID and FACEBOOK_PAGE_ACCESS_TOKEN to .env"}

        try:
            if is_reel:
                # Facebook Video Reels API (3-phase upload)
                print(f"  [Facebook] Initiating Reels upload for '{title}'...")
                # Phase 1: Start
                start_url = f"{self.GRAPH_BASE}/{self.page_id}/video_reels"
                start_resp = requests.post(
                    start_url,
                    data={"upload_phase": "start", "access_token": self.access_token},
                    timeout=30,
                )
                if start_resp.status_code != 200:
                    # Fallback to standard Page video upload if Reels endpoint isn't enabled for the page
                    print(f"  [Facebook] Reels endpoint returned {start_resp.status_code}; falling back to standard video API...")
                    return self._upload_standard_video(video_p, title, description)

                start_data = start_resp.json()
                video_id = start_data.get("video_id")
                upload_url = start_data.get("upload_url")

                if not (video_id and upload_url):
                    return self._upload_standard_video(video_p, title, description)

                # Phase 2: Binary Upload
                print(f"  [Facebook] Transferring video binary ({file_size / (1024*1024):.2f} MB)...")
                with open(video_p, "rb") as f:
                    up_headers = {
                        "Authorization": f"OAuth {self.access_token}",
                        "offset": "0",
                        "file_size": str(file_size),
                    }
                    up_resp = requests.post(upload_url, headers=up_headers, data=f, timeout=600)

                if up_resp.status_code not in (200, 201):
                    return {"success": False, "error": f"Facebook binary transfer failed: {up_resp.text}"}

                # Phase 3: Finish & Publish
                print(f"  [Facebook] Publishing Reel...")
                finish_resp = requests.post(
                    f"{self.GRAPH_BASE}/{self.page_id}/video_reels",
                    data={
                        "upload_phase": "finish",
                        "video_id": video_id,
                        "video_state": "PUBLISHED",
                        "description": description,
                        "title": title[:255],
                        "access_token": self.access_token,
                    },
                    timeout=45,
                )
                if finish_resp.status_code == 200:
                    fb_url = f"https://www.facebook.com/reel/{video_id}"
                    print(f"  [Facebook] ✓ Facebook Reel published successfully! URL: {fb_url}")
                    return {
                        "success": True,
                        "platform": "facebook",
                        "video_id": video_id,
                        "url": fb_url,
                        "type": "reel",
                    }
                else:
                    return {"success": False, "error": f"Facebook reel publishing failed: {finish_resp.text}"}
            else:
                return self._upload_standard_video(video_p, title, description)

        except Exception as e:
            return {"success": False, "error": f"Facebook upload exception: {e}"}

    def _upload_standard_video(self, video_p: Path, title: str, description: str) -> dict[str, Any]:
        """Upload standard page video via POST /{page_id}/videos."""
        print(f"  [Facebook] Uploading standard Page video to Facebook...")
        url = f"{self.GRAPH_BASE}/{self.page_id}/videos"
        with open(video_p, "rb") as vf:
            files = {"source": (video_p.name, vf, "video/mp4")}
            data = {
                "title": title[:255],
                "description": description,
                "published": "true",
                "access_token": self.access_token,
            }
            resp = requests.post(url, files=files, data=data, timeout=600)
        if resp.status_code == 200:
            res = resp.json()
            vid = res.get("id")
            fb_url = f"https://www.facebook.com/{self.page_id}/videos/{vid}"
            print(f"  [Facebook] ✓ Facebook Video published! URL: {fb_url}")
            return {"success": True, "platform": "facebook", "video_id": vid, "url": fb_url, "type": "video"}
        return {"success": False, "error": f"Standard video upload failed ({resp.status_code}): {resp.text}"}

    def upload_reel(
        self,
        video_path: str | Path,
        fb_metadata: dict | None = None,
        dry_run: bool = False,
        meta: dict | None = None,
    ) -> dict[str, Any]:
        """Convenience method to upload video as a Facebook Reel."""
        return self.upload_video(
            video_path=video_path,
            fb_metadata=fb_metadata or meta or {},
            is_reel=True,
            dry_run=dry_run,
        )


# --------------------------------------------------------------------------
# Platform Uploaders: Instagram
# --------------------------------------------------------------------------

class InstagramUploader:
    """Meta Instagram Content Publishing API for Instagram Business/Creator Reels."""

    GRAPH_BASE = "https://graph.facebook.com/v19.0"

    def __init__(
        self,
        user_id: str | None = None,
        access_token: str | None = None,
    ) -> None:
        self.user_id = user_id or os.environ.get("INSTAGRAM_USER_ID") or os.environ.get("IG_USER_ID") or os.environ.get("IG_BUSINESS_ACCOUNT_ID")
        self.access_token = access_token or os.environ.get("INSTAGRAM_ACCESS_TOKEN") or os.environ.get("IG_ACCESS_TOKEN")

    def is_configured(self) -> bool:
        """Return True if Instagram User ID and Access Token are available."""
        return bool(self.user_id and self.access_token)

    def check_connection(self) -> dict[str, Any]:
        """Verify Instagram Business account connectivity and access token."""
        if not self.is_configured():
            return {"configured": False, "status": "missing_credentials", "message": "No INSTAGRAM_USER_ID or INSTAGRAM_ACCESS_TOKEN found in .env"}

        url = f"{self.GRAPH_BASE}/{self.user_id}?fields=id,username,name,profile_picture_url&access_token={self.access_token}"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                username = data.get("username", "instagram_user")
                return {
                    "configured": True,
                    "status": "connected",
                    "user_id": data.get("id"),
                    "username": username,
                    "message": f"Connected to Instagram Account '@{username}' (ID: {self.user_id})",
                }
            return {"configured": True, "status": "error", "code": resp.status_code, "message": resp.text[:200]}
        except Exception as e:
            return {"configured": True, "status": "network_error", "message": str(e)}

    def upload_reel(
        self,
        video_path: str | Path,
        ig_metadata: dict | None = None,
        video_url: str | None = None,
        dry_run: bool = False,
        meta: dict | None = None,
    ) -> dict[str, Any]:
        """Create Instagram Reel media container, upload video, wait for ready, and publish."""
        video_p = Path(video_path).resolve()
        if not dry_run and not video_p.exists() and not video_url:
            return {"success": False, "error": f"Video file not found: {video_p}"}

        ig_meta = ig_metadata or meta or {}
        caption = ig_meta.get("caption", "")
        file_size = video_p.stat().st_size if video_p.exists() else 0

        if dry_run:
            print(f"  [Instagram] [DRY RUN] Would upload Reel to @{self.user_id or 'account'}")
            print(f"              Caption: {caption[:80]}...")
            print(f"              Hashtags: {ig_meta.get('hashtags_count', 0)} tags included")
            return {
                "success": True,
                "platform": "instagram",
                "id": "DRY_RUN_IG_ID",
                "media_id": "DRY_RUN_IG_ID",
                "url": "https://www.instagram.com/reel/DRY_RUN_IG_ID",
                "status": "dry_run_simulated",
            }

        if not self.is_configured():
            return {"success": False, "error": "Instagram credentials missing. Set INSTAGRAM_USER_ID and INSTAGRAM_ACCESS_TOKEN in .env"}

        try:
            # Step 1: Create Reel Container
            print(f"  [Instagram] Creating Reel container...")
            create_url = f"{self.GRAPH_BASE}/{self.user_id}/media"
            
            # Use public video URL if provided or resumable binary container
            if video_url:
                payload = {
                    "media_type": "REELS",
                    "video_url": video_url,
                    "caption": caption,
                    "share_to_feed": "true",
                    "access_token": self.access_token,
                }
                c_resp = requests.post(create_url, data=payload, timeout=45)
            else:
                payload = {
                    "media_type": "REELS",
                    "upload_type": "resumable",
                    "caption": caption,
                    "share_to_feed": "true",
                    "access_token": self.access_token,
                }
                c_resp = requests.post(create_url, data=payload, timeout=45)

            if c_resp.status_code != 200:
                return {"success": False, "error": f"Container creation failed ({c_resp.status_code}): {c_resp.text}"}

            c_data = c_resp.json()
            container_id = c_data.get("id")
            upload_uri = c_data.get("uri")

            if not container_id:
                return {"success": False, "error": f"No container ID returned: {c_data}"}

            # Step 2: If resumable binary upload, transfer video bytes
            if upload_uri and video_p.exists():
                print(f"  [Instagram] Uploading video binary ({file_size / (1024*1024):.2f} MB)...")
                with open(video_p, "rb") as vf:
                    up_headers = {
                        "Authorization": f"OAuth {self.access_token}",
                        "offset": "0",
                        "file_size": str(file_size),
                    }
                    up_resp = requests.post(upload_uri, headers=up_headers, data=vf, timeout=600)
                if up_resp.status_code not in (200, 201):
                    return {"success": False, "error": f"Instagram binary transfer failed: {up_resp.text}"}

            # Step 3: Poll status until READY / FINISHED
            print(f"  [Instagram] Processing video on Meta servers (container {container_id})...")
            status_url = f"{self.GRAPH_BASE}/{container_id}?fields=status_code,status&access_token={self.access_token}"
            max_attempts = 30
            for attempt in range(max_attempts):
                time.sleep(4)
                s_resp = requests.get(status_url, timeout=15)
                if s_resp.status_code == 200:
                    s_data = s_resp.json()
                    status_code = s_data.get("status_code", "")
                    if status_code in ("FINISHED", "READY"):
                        print(f"  [Instagram] Video processing complete!")
                        break
                    elif status_code == "ERROR":
                        return {"success": False, "error": f"Instagram video processing failed: {s_data.get('status')}"}
                    print(f"  [Instagram] Status: {status_code} (waiting...)")
                else:
                    print(f"  [Instagram] Status check warning ({s_resp.status_code}): {s_resp.text[:100]}")

            # Step 4: Publish Reel
            print(f"  [Instagram] Publishing Reel to feed...")
            pub_url = f"{self.GRAPH_BASE}/{self.user_id}/media_publish"
            pub_resp = requests.post(
                pub_url,
                data={"creation_id": container_id, "access_token": self.access_token},
                timeout=45,
            )
            if pub_resp.status_code == 200:
                media_id = pub_resp.json().get("id")
                ig_url = f"https://www.instagram.com/reel/{media_id}/"
                print(f"  [Instagram] ✓ Reel published successfully! Media ID: {media_id}")
                return {
                    "success": True,
                    "platform": "instagram",
                    "media_id": media_id,
                    "url": ig_url,
                }
            else:
                return {"success": False, "error": f"Publishing failed ({pub_resp.status_code}): {pub_resp.text}"}

        except Exception as e:
            return {"success": False, "error": f"Instagram upload exception: {e}"}


# --------------------------------------------------------------------------
# Multi-Platform Orchestrator & CLI
# --------------------------------------------------------------------------

def upload_all(
    video_path: str | Path,
    metadata_path: str | Path | None = None,
    platforms: list[str] | None = None,
    privacy: str = "public",
    publish_at: str | None = None,
    srt_path: str | Path | None = None,
    thumbnail_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Upload a video across requested platforms (YouTube, Facebook, Instagram)."""
    video_p = Path(video_path).resolve()
    if not dry_run and not video_p.exists():
        print(f"Error: Video file not found at {video_p}", file=sys.stderr)
        return {"success": False, "error": f"Video not found: {video_p}"}

    # Resolve metadata JSON
    meta_data = {}
    if metadata_path and Path(metadata_path).exists():
        try:
            meta_data = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        except Exception:
            pass
    elif (ROOT / "out" / "metadata.json").exists():
        try:
            meta_data = json.loads((ROOT / "out" / "metadata.json").read_text(encoding="utf-8"))
        except Exception:
            pass

    # If no metadata file found, generate on the fly from script.json
    if not meta_data and (ROOT / "script.json").exists():
        try:
            import metadata
            script_raw = json.loads((ROOT / "script.json").read_text(encoding="utf-8"))
            meta_data = metadata.generate_metadata(script_raw)
        except Exception as e:
            print(f"Note: Could not load metadata: {e}", file=sys.stderr)

    # Auto-detect subtitles & thumbnails if not provided
    if not srt_path:
        for candidate in [ROOT / "out" / "subtitles.srt", ROOT / "src" / "generated" / "subtitles.srt"]:
            if candidate.exists():
                srt_path = candidate
                break

    if not thumbnail_path:
        for candidate in [ROOT / "out" / "thumbnail.jpg", ROOT / "public" / "images" / "scene-001.png"]:
            if candidate.exists():
                thumbnail_path = candidate
                break

    target_platforms = [p.lower().strip() for p in (platforms or ["youtube", "facebook", "instagram"])]
    if "all" in target_platforms:
        target_platforms = ["youtube", "facebook", "instagram"]

    print("\n" + "=" * 74)
    print("MULTI-PLATFORM VIDEO AUTO-UPLOADER")
    print("=" * 74)
    print(f"Video File : {video_p}")
    file_mb = (video_p.stat().st_size / (1024*1024)) if video_p.exists() else 0.0
    print(f"File Size  : {file_mb:.2f} MB")
    print(f"Platforms  : {', '.join(target_platforms).upper()}")
    print(f"Privacy    : {privacy.upper()}")
    if dry_run:
        print("Mode       : [DRY RUN / SIMULATION]")
    print("-" * 74)

    results: dict[str, Any] = {
        "timestamp": time.time(),
        "video": str(video_p),
        "platforms": {},
    }

    # 1. YouTube Upload
    if "youtube" in target_platforms or "yt" in target_platforms:
        yt = YouTubeUploader()
        yt_meta = meta_data.get("youtube", {
            "title": meta_data.get("title", video_p.stem),
            "description": meta_data.get("description", ""),
            "tags": meta_data.get("tags", []),
        })
        res = yt.upload_video(
            video_p,
            yt_meta,
            privacy_status=privacy,
            publish_at=publish_at,
            srt_path=srt_path,
            thumbnail_path=thumbnail_path,
            dry_run=dry_run,
        )
        results["platforms"]["youtube"] = res

    # 2. Facebook Upload
    if "facebook" in target_platforms or "fb" in target_platforms:
        fb = FacebookUploader()
        fb_meta = meta_data.get("facebook", {
            "title": meta_data.get("title", video_p.stem),
            "description": meta_data.get("description", ""),
        })
        is_reel = meta_data.get("format") in ("reel", "portrait", "square") or "reel" in video_p.stem.lower()
        res = fb.upload_video(
            video_p,
            fb_meta,
            is_reel=is_reel,
            dry_run=dry_run,
        )
        results["platforms"]["facebook"] = res

    # 3. Instagram Upload
    if "instagram" in target_platforms or "ig" in target_platforms:
        ig = InstagramUploader()
        ig_meta = meta_data.get("instagram", {
            "caption": meta_data.get("description", ""),
        })
        res = ig.upload_reel(
            video_p,
            ig_meta,
            dry_run=dry_run,
        )
        results["platforms"]["instagram"] = res

    # Summary Report
    print("\n" + "=" * 74)
    print("UPLOAD SUMMARY")
    print("=" * 74)
    for p_name, p_res in results["platforms"].items():
        status_sym = "✓ SUCCESS" if p_res.get("success") else "✗ FAILED"
        print(f"  {p_name.upper():<12} : {status_sym}")
        if p_res.get("success"):
            print(f"                 URL : {p_res.get('url', 'N/A')}")
        else:
            print(f"                 Error : {p_res.get('error', 'Unknown error')}")

    # Export results JSON
    report_p = ROOT / "out" / "upload_report.json"
    report_p.parent.mkdir(parents=True, exist_ok=True)
    report_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReport written to: {report_p.relative_to(ROOT)}")
    return results


def check_all_credentials() -> None:
    """Check connectivity and credentials for YouTube, Facebook, and Instagram."""
    print("=" * 74)
    print("CHECKING SOCIAL MEDIA CREDENTIALS & CONNECTIVITY")
    print("=" * 74)

    yt = YouTubeUploader()
    fb = FacebookUploader()
    ig = InstagramUploader()

    # YouTube
    print("\n[1] YouTube (YouTube Data API v3):")
    yt_status = yt.check_connection()
    if yt_status.get("status") == "connected":
        print(f"    ✓ {yt_status['message']}")
    else:
        print(f"    ✗ Status: {yt_status.get('status')} - {yt_status.get('message')}")
        print("      Setup: Set YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET in .env, then run:")
        print("             .venv\\Scripts\\python.exe scripts\\uploader.py --auth youtube")

    # Facebook
    print("\n[2] Facebook (Meta Graph API):")
    fb_status = fb.check_connection()
    if fb_status.get("status") == "connected":
        print(f"    ✓ {fb_status['message']}")
    else:
        print(f"    ✗ Status: {fb_status.get('status')} - {fb_status.get('message')}")
        print("      Setup: Set FACEBOOK_PAGE_ID and FACEBOOK_PAGE_ACCESS_TOKEN in .env")

    # Instagram
    print("\n[3] Instagram (Content Publishing API):")
    ig_status = ig.check_connection()
    if ig_status.get("status") == "connected":
        print(f"    ✓ {ig_status['message']}")
    else:
        print(f"    ✗ Status: {ig_status.get('status')} - {ig_status.get('message')}")
        print("      Setup: Set INSTAGRAM_USER_ID and INSTAGRAM_ACCESS_TOKEN in .env")

    print("\n" + "=" * 74)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("video", nargs="?", default=None, help="path to video file (defaults to out/video.mp4)")
    parser.add_argument(
        "--video",
        dest="video_opt",
        default=None,
        help="path to video file (alias for positional video argument)",
    )
    parser.add_argument(
        "--platforms",
        "-p",
        default="youtube,facebook,instagram",
        help="comma-separated platforms to upload to: youtube, facebook, instagram, or all",
    )
    parser.add_argument(
        "--metadata",
        "-m",
        default=None,
        help="path to metadata.json file (defaults to out/metadata.json)",
    )
    parser.add_argument(
        "--privacy",
        choices=["public", "unlisted", "private"],
        default="public",
        help="privacy status for YouTube (public, unlisted, private; default: public)",
    )
    parser.add_argument(
        "--publish-at",
        dest="publish_at",
        default=None,
        help="scheduled publish time in ISO 8601 format (e.g. 2026-10-04T12:00:00Z)",
    )
    parser.add_argument(
        "--subtitles",
        default=None,
        help="path to .srt subtitles file for YouTube captions track",
    )
    parser.add_argument(
        "--thumbnail",
        default=None,
        help="path to custom thumbnail image for YouTube",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="simulate upload process without making actual API calls",
    )
    parser.add_argument(
        "--check",
        "--check-credentials",
        action="store_true",
        help="verify credentials and connectivity for all platforms, then exit",
    )
    parser.add_argument(
        "--auth",
        choices=["youtube", "yt"],
        help="start interactive OAuth2 login flow for the specified platform",
    )
    args = parser.parse_args()

    if args.check:
        check_all_credentials()
        return 0

    if args.auth in ("youtube", "yt"):
        yt = YouTubeUploader()
        success = yt.start_oauth_flow()
        return 0 if success else 1

    # Locate video file
    video_file = args.video_opt or args.video
    if not video_file:
        for candidate in [ROOT / "out" / "video.mp4", ROOT / "out" / "imagetext.mp4", ROOT / "out" / "mobilephonescamalert.mp4"]:
            if candidate.exists():
                video_file = str(candidate)
                break

    if not video_file:
        mp4_files = sorted((ROOT / "out").glob("*.mp4"))
        if mp4_files:
            video_file = str(mp4_files[0])

    if not video_file:
        print("Error: No video file found. Please specify video path or render one first with 'npm run render'", file=sys.stderr)
        return 1

    platforms_list = [p.strip() for p in args.platforms.split(",") if p.strip()]
    res = upload_all(
        video_path=video_file,
        metadata_path=args.metadata,
        platforms=platforms_list,
        privacy=args.privacy,
        publish_at=args.publish_at,
        srt_path=args.subtitles,
        thumbnail_path=args.thumbnail,
        dry_run=args.dry_run,
    )

    all_ok = any(p_res.get("success") for p_res in res.get("platforms", {}).values())
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
