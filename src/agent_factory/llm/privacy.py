"""Privacy modes and the local-only guard.

In `local` mode nothing may leave this machine or the local network: only Ollama models that run
locally, an LLMP address on loopback or a private network, and no prompt text in LLMP's call log.
The checks run before any request is made, so a misconfigured run fails without sending anything.
"""

from __future__ import annotations

import ipaddress
import socket
from enum import StrEnum
from urllib.parse import urlsplit


class PrivacyMode(StrEnum):
    LOCAL = "local"
    CLOUD_OK = "cloud_ok"


class PrivacyError(RuntimeError):
    """A run would send data somewhere its privacy mode does not allow."""


# LLMP's `llmp.log` setting per mode: "meta" keeps prompts and replies out of chat_log.
LLMP_LOG_MODE = {PrivacyMode.LOCAL: "meta", PrivacyMode.CLOUD_OK: None}


def is_ollama_cloud(model: str) -> bool:
    """Ollama's cloud models (`:cloud`, `-cloud` tags) run on ollama.com even through a local Ollama."""
    name = model.removeprefix("ollama/")
    base, _, tag = name.partition(":")
    return tag == "cloud" or tag.endswith("-cloud") or base.endswith("-cloud")


def check_model(model: str, mode: PrivacyMode) -> None:
    if mode is PrivacyMode.CLOUD_OK:
        return
    if not model.startswith("ollama/"):
        raise PrivacyError(
            f"model {model!r} is not local: privacy mode 'local' allows only ollama/ models"
        )
    if is_ollama_cloud(model):
        raise PrivacyError(
            f"model {model!r} is an Ollama cloud model and runs on ollama.com: "
            "not allowed in privacy mode 'local'"
        )


def check_base_url(base_url: str, mode: PrivacyMode) -> None:
    """In local mode, every address the LLMP host resolves to must be loopback or private."""
    if mode is PrivacyMode.CLOUD_OK:
        return
    host = urlsplit(base_url).hostname
    if not host:
        raise PrivacyError(f"cannot read a host from the LLMP address {base_url!r}")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise PrivacyError(f"cannot resolve the LLMP host {host!r}: {e}") from e
    for *_, sockaddr in infos:
        address = ipaddress.ip_address(sockaddr[0])
        if not (address.is_loopback or address.is_private):
            raise PrivacyError(
                f"LLMP host {host!r} resolves to {address}, which is not on this machine or a "
                "private network: not allowed in privacy mode 'local'"
            )


def effective_mode(
    requested: PrivacyMode | None, scenario_mode: PrivacyMode, sensitive: bool
) -> PrivacyMode:
    """The mode a run uses: the request, else the scenario's, and always local with sensitive data."""
    mode = requested or scenario_mode
    if sensitive and mode is PrivacyMode.CLOUD_OK:
        raise PrivacyError(
            "this scenario uses a profile marked sensitive, which forces privacy mode 'local'"
        )
    return mode
