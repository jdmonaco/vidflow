"""Video metadata protocol for abstracting different video sources."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class VideoMetadataProtocol(Protocol):
    """Protocol defining required metadata fields for any video source."""

    @property
    def identifier(self) -> str: ...

    @property
    def title(self) -> str: ...

    @property
    def author(self) -> str | None: ...

    @property
    def source_date(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def duration(self) -> float: ...

    @property
    def source_type(self) -> str: ...
