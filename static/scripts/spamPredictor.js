const handleSubmit = async (e) => {
  e.preventDefault()
  const text = document.querySelector("#email-input").textContent
  const req = await fetch('/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ document: text, redirect: false })
  })
  const data = await req.json()
  document.querySelector("#results").style = "display: block;"
  document.querySelector("#result").textContent = data.classification
  document.querySelector("#confidence").textContent = data.confidence
}
const form = document.querySelector("#data-form")
form.addEventListener("submit", handleSubmit)