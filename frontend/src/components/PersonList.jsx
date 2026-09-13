import { useState } from 'react'

export default function PersonList({ people, loading, onRemove }) {
  const [removing, setRemoving] = useState(null)

  const handleRemove = async (name) => {
    if (removing) return
    setRemoving(name)
    try {
      await onRemove(name)
    } finally {
      setRemoving(null)
    }
  }

  return (
    <section className="card">
      <div className="card-head">
        <h2 className="card-title">Enrolled people</h2>
        {!loading && people.length > 0 && (
          <span className="card-count">{people.length}</span>
        )}
      </div>

      {loading ? (
        <p className="empty">Loading…</p>
      ) : people.length === 0 ? (
        <p className="empty">No people enrolled yet. Add someone using the form below.</p>
      ) : (
        <ul className="person-list">
          {people.map(name => (
            <li key={name} className="person-row">
              <span className="person-name" title={name}>{name}</span>
              <button
                type="button"
                className={`btn-text${removing === name ? ' is-busy' : ''}`}
                onClick={() => handleRemove(name)}
                disabled={removing !== null}
              >
                {removing === name ? 'Removing…' : 'Remove'}
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
