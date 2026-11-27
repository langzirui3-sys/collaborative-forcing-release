"""Worker entry for the explicitly tracked, otherwise unchanged SDVG backend."""
from . import accepted_stream_worker as stream
from .tracked_sdvg import observed_worker_class


def main(argv=None):
    original = stream.StreamingCorrectionWorker
    stream.StreamingCorrectionWorker = observed_worker_class(original)
    try:
        return stream.main(argv)
    finally:
        stream.StreamingCorrectionWorker = original


if __name__ == '__main__':
    raise SystemExit(main())
