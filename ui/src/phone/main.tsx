import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './phone.css'
import PhoneSender from './PhoneSender.tsx'

// phone.html: the phone camera page, separate from the console so the phone loads nothing else.
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <PhoneSender />
  </StrictMode>,
)
