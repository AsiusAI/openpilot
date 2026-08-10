#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import uuid
from typing import Any

import jwt
import requests
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat

from openpilot.system.app.identity import bytes_to_identity, identity_to_bytes

PROTOCOL_VERSION = 1
ENCRYPTION_MAGIC = b"ASIUSDATA1\n"


def wall_time() -> float:
  return time.time()  # noqa: TID251


def b64url(value: bytes) -> str:
  return base64.urlsafe_b64encode(value).decode().rstrip("=")


def canonical_json(value: Any) -> str:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def public_identity(private_key: ed25519.Ed25519PrivateKey) -> str:
  return bytes_to_identity(private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw))


def x25519_private(private_key: ed25519.Ed25519PrivateKey) -> x25519.X25519PrivateKey:
  raw = private_key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
  digest = bytearray(hashlib.sha512(raw).digest()[:32])
  digest[0] &= 248
  digest[31] &= 127
  digest[31] |= 64
  return x25519.X25519PrivateKey.from_private_bytes(bytes(digest))


def x25519_public(identity: str) -> x25519.X25519PublicKey:
  p = 2**255 - 19
  raw = bytearray(identity_to_bytes(identity))
  raw[31] &= 0x7f
  y = int.from_bytes(raw, "little")
  u = ((1 + y) * pow(1 - y, p - 2, p)) % p
  return x25519.X25519PublicKey.from_public_bytes(u.to_bytes(32, "little"))


def can_wrap_for(private_key: ed25519.Ed25519PrivateKey, identity: str) -> bool:
  try:
    x25519_private(private_key).exchange(x25519_public(identity))
    return True
  except (TypeError, ValueError):
    return False


def wrap_folder_key(private_key: ed25519.Ed25519PrivateKey, owner: str, key_id: str, recipient: str, folder_key: bytes) -> dict[str, Any]:
  shared = x25519_private(private_key).exchange(x25519_public(recipient))
  iv = os.urandom(12)
  aad = {"v": 1, "alg": "Ed25519-X25519-A256GCM", "from": owner, "to": recipient, "owner": owner, "keyId": key_id, "iv": b64url(iv)}
  wrap_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=f"asius-data-access-v1:{owner}:{key_id}".encode(),
                  info=f"{owner}:{recipient}".encode()).derive(shared)
  ciphertext = AESGCM(wrap_key).encrypt(iv, folder_key, canonical_json(aad).encode())
  return {key: value for key, value in aad.items() if key != "owner"} | {"ciphertext": b64url(ciphertext)}


def access_document(private_key: ed25519.Ed25519PrivateKey, state: dict[str, Any], readers: list[str]) -> dict[str, Any]:
  owner = public_identity(private_key)
  recipients = list(dict.fromkeys([owner, *readers]))
  keys = []
  for entry in state["keys"]:
    raw_key = base64.urlsafe_b64decode(entry["key"] + "=" * (-len(entry["key"]) % 4))
    keys.append({
      "id": entry["id"],
      "createdAt": entry["createdAt"],
      "grants": [wrap_folder_key(private_key, owner, entry["id"], recipient, raw_key) for recipient in recipients],
    })
  unsigned = {
    "v": 1,
    "owner": owner,
    "version": state["version"],
    "updatedAt": int(wall_time() * 1000),
    "activeKey": state["activeKey"],
    "keys": keys,
  }
  return unsigned | {"signature": b64url(private_key.sign(canonical_json(unsigned).encode()))}


def recover_state(private_key: ed25519.Ed25519PrivateKey, document: dict[str, Any], readers: list[str]) -> dict[str, Any]:
  owner = public_identity(private_key)
  if document.get("owner") != owner:
    raise ValueError("access document owner mismatch")
  unsigned = {key: value for key, value in document.items() if key != "signature"}
  private_key.public_key().verify(base64.urlsafe_b64decode(document["signature"] + "=" * (-len(document["signature"]) % 4)), canonical_json(unsigned).encode())
  recovered = []
  for entry in document["keys"]:
    grant = next(grant for grant in entry["grants"] if grant["to"] == owner)
    shared = x25519_private(private_key).exchange(x25519_public(grant["from"]))
    wrap_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=f"asius-data-access-v1:{owner}:{entry['id']}".encode(),
                    info=f"{grant['from']}:{owner}".encode()).derive(shared)
    aad = {"v": grant["v"], "alg": grant["alg"], "from": grant["from"], "to": owner, "owner": owner, "keyId": entry["id"], "iv": grant["iv"]}
    iv = base64.urlsafe_b64decode(grant["iv"] + "=" * (-len(grant["iv"]) % 4))
    ciphertext = base64.urlsafe_b64decode(grant["ciphertext"] + "=" * (-len(grant["ciphertext"]) % 4))
    folder_key = AESGCM(wrap_key).decrypt(iv, ciphertext, canonical_json(aad).encode())
    recovered.append({"id": entry["id"], "createdAt": entry["createdAt"], "key": b64url(folder_key)})
  return {"version": document["version"] + 1, "activeKey": document["activeKey"], "readers": readers, "keys": recovered}


def initial_state(readers: list[str]) -> dict[str, Any]:
  key_id = f"k_{b64url(os.urandom(18))}"
  return {
    "version": 1,
    "activeKey": key_id,
    "readers": readers,
    "keys": [{"id": key_id, "createdAt": int(wall_time() * 1000), "key": b64url(os.urandom(32))}],
  }


def update_state(state: dict[str, Any], readers: list[str]) -> dict[str, Any]:
  previous = set(state.get("readers", []))
  current = set(readers)
  if previous == current:
    return state
  state = json.loads(json.dumps(state))
  state["version"] += 1
  state["readers"] = readers
  if previous - current:
    key_id = f"k_{b64url(os.urandom(18))}"
    state["activeKey"] = key_id
    state["keys"].append({"id": key_id, "createdAt": int(wall_time() * 1000), "key": b64url(os.urandom(32))})
  return state


def encrypt_object(owner: str, path: str, plaintext: bytes, state: dict[str, Any]) -> bytes:
  active = next(entry for entry in state["keys"] if entry["id"] == state["activeKey"])
  folder_key = base64.urlsafe_b64decode(active["key"] + "=" * (-len(active["key"]) % 4))
  iv = os.urandom(12)
  header = {"v": 1, "alg": "HKDF-SHA256-A256GCM", "owner": owner, "path": path, "keyId": active["id"], "iv": b64url(iv), "plaintextLength": len(plaintext)}
  object_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=f"asius-data-object-v1:{owner}:{active['id']}".encode(), info=path.encode()).derive(folder_key)
  ciphertext = AESGCM(object_key).encrypt(iv, plaintext, canonical_json(header).encode())
  encoded_header = json.dumps(header, separators=(",", ":")).encode()
  return ENCRYPTION_MAGIC + len(encoded_header).to_bytes(4, "big") + encoded_header + ciphertext


class DataApiClient:
  def __init__(self, base_url: str, private_key: ed25519.Ed25519PrivateKey, session: requests.Session | None = None):
    self.base_url = base_url.rstrip("/")
    self.private_key = private_key
    self.owner = public_identity(private_key)
    self.session = session or requests.Session()
    self._config: dict[str, Any] | None = None

  def request(self, method: str, path: str, body: Any = None) -> requests.Response:
    encoded = b"" if body is None else canonical_json(body).encode()
    now = int(wall_time())
    claims = {
      "v": 1,
      "identity": self.owner,
      "method": method,
      "path": path,
      "bodyHash": b64url(hashlib.sha256(encoded).digest()),
      "nonce": str(uuid.uuid4()),
      "iat": now,
      "nbf": now,
      "exp": now + 300,
    }
    token = jwt.encode(claims, self.private_key, algorithm="EdDSA")
    headers = {"Authorization": f"Data {token}", **({"Content-Type": "application/json"} if encoded else {})}
    response = self.session.request(method, self.base_url + path, data=encoded or None, headers=headers, timeout=30)
    response.raise_for_status()
    return response

  def put_access(self, document: dict[str, Any]) -> None:
    self.request("PUT", f"/v1/{self.owner}/access", document)

  def get_access(self) -> dict[str, Any]:
    return self.request("GET", f"/v1/{self.owner}/access").json()

  def get_config(self) -> dict[str, Any]:
    if self._config is None:
      response = self.session.get(f"{self.base_url}/v1/config", timeout=10)
      response.raise_for_status()
      self._config = response.json()
    return self._config

  def register_upload(self, path: str, *, content_type: str, content_length: int, plaintext_length: int,
                      route_start_time: int, checksum: str, media: dict[str, Any] | None = None) -> dict[str, Any]:
    request_path = f"/v1/{self.owner}/uploads"
    body = {
      "path": path,
      "contentType": content_type,
      "contentLength": content_length,
      "plaintextLength": plaintext_length,
      "routeStartTime": route_start_time,
      "checksumSha256": checksum,
      **({"media": media} if media is not None else {}),
    }
    return self.request("POST", request_path, body).json()

  def complete_upload(self, path: str, checksum: str) -> None:
    self.request("POST", f"/v1/{self.owner}/uploads/complete", {"path": path, "checksumSha256": checksum})

  def upload(self, path: str, encrypted: bytes, *, plaintext_length: int, route_start_time: int) -> requests.Response:
    checksum = b64url(hashlib.sha256(encrypted).digest())
    signed = self.register_upload(path, content_type="application/octet-stream", content_length=len(encrypted),
                                  plaintext_length=plaintext_length, route_start_time=route_start_time, checksum=checksum)
    response = self.session.put(signed["url"], data=encrypted, headers=signed["headers"], timeout=120)
    response.raise_for_status()
    self.complete_upload(path, checksum)
    return response

  def upload_file(self, path: str, filename: str, *, content_type: str, plaintext_length: int,
                  route_start_time: int, checksum: str, media: dict[str, Any]) -> requests.Response:
    content_length = os.path.getsize(filename)
    signed = self.register_upload(path, content_type=content_type, content_length=content_length,
                                  plaintext_length=plaintext_length, route_start_time=route_start_time,
                                  checksum=checksum, media=media)
    with open(filename, "rb") as stream:
      response = self.session.put(signed["url"], data=stream, headers=signed["headers"], timeout=120)
    response.raise_for_status()
    self.complete_upload(path, checksum)
    return response
