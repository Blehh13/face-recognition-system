import { useState } from 'react'
import { ROUTES, navigate } from '../router.js'

/**
 * Who the system knows.
 *
 * This is the home screen because it answers the question everything else
 * depends on: identification is meaningless against an empty directory, and
 * the old single-dashboard layout let you sit in that state without noticing.
 */
export default function Directory({ people, loadingPeople, onRemove }) {
  const [removing, setRemoving] = useState(null)
  const [confirming, setConfirming] = useState(null)

  const remove = async (name) => {
    setRemoving(name)
    setConfirming(null)
    try {
      await onRemove(name)
    } finally {
      setRemoving(null)
    }
  }

  if (loadingPeople) {
    return (
      <div className="view">
        <p className="muted">Loading…</p>
      </div>
    )
  }

  if (people.length === 0) {
    return (
      <div className="view">
        <div className="onboard">
          <p className="onboard-step">Step 1 of 2</p>
          <h1 className="onboard-title">Nobody is enrolled yet</h1>
          <p className="onboard-body">
            The system can only recognise people it has been shown. Add a first
            person, then it will have something to compare new photos against.
          </p>
          <button type="button" className="btn btn-lg" onClick={() => navigate(ROUTES.ADD)}>
            Add the first person
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="view">
      <header className="view-head">
        <div>
          <h1 className="view-title">People</h1>
          <p className="view-sub">
            {people.length === 1 ? '1 person' : `${people.length} people`} the system can recognise
          </p>
        </div>
        <div className="view-actions">
          <button type="button" className="btn btn-ghost" onClick={() => navigate(ROUTES.ADD)}>
            Add someone
          </button>
          <button type="button" className="btn" onClick={() => navigate(ROUTES.IDENTIFY)}>
            Identify a face
          </button>
        </div>
      </header>

      <ul className="people">
        {people.map(name => (
          <li key={name} className="person">
            <span className="person-initial" aria-hidden="true">
              {name.trim().charAt(0).toUpperCase()}
            </span>
            <span className="person-name" title={name}>{name}</span>

            {confirming === name ? (
              <span className="person-confirm">
                <span className="person-confirm-text">Remove?</span>
                <button
                  type="button"
                  className="btn-danger-sm"
                  onClick={() => remove(name)}
                  disabled={removing === name}
                >
                  {removing === name ? 'Removing…' : 'Yes, remove'}
                </button>
                <button type="button" className="btn-link" onClick={() => setConfirming(null)}>
                  Cancel
                </button>
              </span>
            ) : (
              <button
                type="button"
                className="person-remove"
                onClick={() => setConfirming(name)}
                disabled={removing !== null}
              >
                Remove
              </button>
            )}
          </li>
        ))}
      </ul>
    </div>
  )
}
