import { useState, useRef, useEffect } from 'react'
import CameraCapture from './CameraCapture.jsx'

export default function EnrollForm({ onEnrolled, onError }) {
  const [name, setName] = useState('')
  const [files, setFiles] = useState([])
  const [previews, setPreviews] = useState([])
  const [dragging, setDragging] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [cameraOpen, setCameraOpen] = useState(false)
  const inputRef = useRef(null)

  useEffect(() => {
    const urls = files.map(file => URL.createObjectURL(file))
    setPreviews(urls)
    return () => urls.forEach(url => URL.revokeObjectURL(url))
  }, [files])

  const acceptFiles = (fileList) => {
    const images = Array.from(fileList).filter(file => file.type.startsWith('image/'))
    if (images.length === 0) {
      onError('Those files are not images. Choose a JPG, PNG, BMP or WebP photo.')
      return
    }
    setFiles(images)
  }

  const clearForm = () => {
    setName('')
    setFiles([])
    if (inputRef.current) inputRef.current.value = ''
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    const trimmed = name.trim()

    if (!trimmed) {
      onError('Enter a name before enrolling.')
      return
    }
    if (files.length === 0) {
      onError('Choose at least one photo before enrolling.')
      return
    }

    setSubmitting(true)
    const body = new FormData()
    body.append('name', trimmed)
    files.forEach(file => body.append('images', file))

    try {
      const res = await fetch('/enroll', {
        method: 'POST',
        headers: { Accept: 'application/json' },
        body,
      })
      const data = await res.json().catch(() => null)

      if (!res.ok || !data) {
        onError(data?.error ?? 'Enrolling failed. Check that the server is running.')
        return
      }

      onEnrolled(data)
      clearForm()
    } catch (err) {
      onError(`Could not reach the server: ${err.message}`)
    } finally {
      setSubmitting(false)
    }
  }

  const fileSummary = files.length === 0
    ? null
    : files.length === 1
      ? files[0].name
      : `${files.length} photos selected`

  return (
    <section className="card">
      <div className="card-head">
        <h2 className="card-title">Add a person</h2>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="field">
          <label className="field-label" htmlFor="enroll-name">Name</label>
          <input
            id="enroll-name"
            className="input"
            type="text"
            value={name}
            onChange={event => setName(event.target.value)}
            placeholder="Alice Johnson"
            disabled={submitting}
            autoComplete="off"
          />
        </div>

        <div className="field">
          <label className="field-label" htmlFor="enroll-files">Photos</label>
          {cameraOpen ? (
            <CameraCapture
              disabled={submitting}
              onCapture={f => { setCameraOpen(false); setFiles([f]) }}
              onError={onError}
              onClose={() => setCameraOpen(false)}
            />
          ) : (
          <button
            type="button"
            id="enroll-files"
            className={`upload${dragging ? ' is-dragging' : ''}`}
            disabled={submitting}
            onClick={() => inputRef.current?.click()}
            onDragOver={event => { event.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={event => {
              event.preventDefault()
              setDragging(false)
              if (submitting) return
              if (event.dataTransfer.files.length) acceptFiles(event.dataTransfer.files)
            }}
          >
            {fileSummary ? (
              <>
                <div className="upload-main">{fileSummary}</div>
                <div className="upload-hint">Click to choose different photos</div>
              </>
            ) : (
              <>
                <div className="upload-arrow" aria-hidden="true">&#8593;</div>
                <div className="upload-main">Drop photos here, or click to choose</div>
                <div className="upload-hint">One clear face per photo. JPG, PNG, BMP or WebP.</div>
              </>
            )}
          </button>
          )}

          {!cameraOpen && (
            <button
              type="button"
              className="btn-link"
              onClick={() => setCameraOpen(true)}
              disabled={submitting}
            >
              Use camera instead
            </button>
          )}

          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            multiple
            hidden
            onChange={event => {
              if (event.target.files.length) acceptFiles(event.target.files)
            }}
          />

          {/*
            previews trails files by one render (the object URLs are built in an
            effect), so only draw once the two agree — otherwise clearing the form
            indexes into an already-empty files array.
          */}
          {files.length > 0 && previews.length === files.length && (
            files.length === 1 ? (
              <img className="preview" src={previews[0]} alt={files[0].name} />
            ) : (
              <div className="thumbs">
                {files.map((file, index) => (
                  <img key={previews[index]} className="thumb" src={previews[index]} alt={file.name} />
                ))}
              </div>
            )
          )}
        </div>

        <button type="submit" className="btn btn-block" disabled={submitting}>
          {submitting ? (
            <>
              <span className="spinner" aria-hidden="true" />
              Enrolling
            </>
          ) : (
            <>
              <span aria-hidden="true">+</span>
              Enroll person
            </>
          )}
        </button>
      </form>
    </section>
  )
}
