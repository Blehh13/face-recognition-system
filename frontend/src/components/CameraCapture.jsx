import { useEffect, useRef, useState } from 'react'

/**
 * Capture a still from the device camera and hand it back as a File, so it
 * travels through exactly the same upload path as a chosen file.
 *
 * The camera is only requested when this component is mounted, and the track
 * is stopped on unmount — a page that silently holds the camera open after
 * you have moved on is a genuine privacy problem, not just untidy.
 */
export default function CameraCapture({ onCapture, onError, onClose, disabled }) {
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const [ready, setReady] = useState(false)
  const [denied, setDenied] = useState(null)

  useEffect(() => {
    let cancelled = false

    async function start() {
      if (!navigator.mediaDevices?.getUserMedia) {
        setDenied('This browser does not support camera capture.')
        return
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
          audio: false,
        })
        if (cancelled) {
          stream.getTracks().forEach(t => t.stop())
          return
        }
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play().catch(() => {})
        }
        setReady(true)
      } catch (err) {
        // Chrome requires a secure context: localhost counts, a LAN IP does not.
        const insecure = window.location.protocol !== 'https:'
          && !['localhost', '127.0.0.1'].includes(window.location.hostname)
        setDenied(
          insecure
            ? 'The camera needs https, or the page opened on localhost.'
            : `Camera unavailable: ${err.name === 'NotAllowedError' ? 'permission denied' : err.message}`
        )
      }
    }

    start()
    return () => {
      cancelled = true
      streamRef.current?.getTracks().forEach(t => t.stop())
      streamRef.current = null
    }
  }, [])

  const capture = () => {
    const video = videoRef.current
    if (!video || !video.videoWidth) {
      onError('The camera is not ready yet.')
      return
    }
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    // The preview is mirrored so it reads like a mirror; undo that for the
    // capture, otherwise every stored face is a flipped version of the person.
    ctx.translate(canvas.width, 0)
    ctx.scale(-1, 1)
    ctx.drawImage(video, 0, 0)

    canvas.toBlob(blob => {
      if (!blob) {
        onError('Could not read a frame from the camera.')
        return
      }
      const stamp = new Date().toISOString().replace(/[:.]/g, '-')
      onCapture(new File([blob], `camera-${stamp}.jpg`, { type: 'image/jpeg' }))
    }, 'image/jpeg', 0.92)
  }

  if (denied) {
    return (
      <div className="camera">
        <p className="camera-error">{denied}</p>
        <button type="button" className="btn-link" onClick={onClose}>
          Use a file instead
        </button>
      </div>
    )
  }

  return (
    <div className="camera">
      <video ref={videoRef} className="camera-feed" playsInline muted />
      <div className="camera-actions">
        <button
          type="button"
          className="btn"
          onClick={capture}
          disabled={disabled || !ready}
        >
          {ready ? 'Take photo' : 'Starting camera…'}
        </button>
        <button type="button" className="btn-link" onClick={onClose} disabled={disabled}>
          Cancel
        </button>
      </div>
    </div>
  )
}
