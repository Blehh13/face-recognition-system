import { useEffect, useState } from 'react'

/**
 * Hash routing in ~30 lines, so the three screens are real destinations.
 *
 * Worth the small effort rather than a `useState` view switch: the back button
 * works, a screen can be linked to, and a reload keeps you where you were.
 * A router library would add a dependency for one file's worth of behaviour.
 */

export const ROUTES = {
  DIRECTORY: '/',
  ADD: '/add',
  IDENTIFY: '/identify',
}

function currentPath() {
  const raw = window.location.hash.replace(/^#/, '')
  return raw || ROUTES.DIRECTORY
}

export function navigate(path) {
  if (currentPath() === path) return
  window.location.hash = path
}

export function useRoute() {
  const [path, setPath] = useState(currentPath)

  useEffect(() => {
    const onChange = () => setPath(currentPath())
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])

  return path
}
