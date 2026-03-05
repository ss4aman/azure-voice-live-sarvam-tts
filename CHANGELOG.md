# Change Log

## [1.0.0] - 2025-07-17

### Added
- Parallel-batch TTS queue — fires all queued sentences simultaneously, delivers audio in order
- Persistent HTTP connection pool for Sarvam API (6 max connections, 4 keepalive)
- Automatic retry on TTS timeout (1 retry)
- Sentence boundary splitting with configurable minimum length
- Ambient audio mixer with office and call center presets
- Web browser test client

### Changed
- Rewrote README for public reference repo (Azure Voice Live + third-party TTS integration)
- Removed hardcoded model defaults from Bicep and server code
- Aligned all Sarvam TTS defaults across infra and application config
- Set `DEBUG_MODE` to `false` for production readiness
- Updated `.env-sample.txt` with safe placeholder values

## [0.0.1] - 2025-07-10

### Added
- Initial commit

