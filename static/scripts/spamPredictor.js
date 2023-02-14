const handleSubmit = async (e) => {
  e.preventDefault()
  const text = document.querySelector("#email-input").textContent
  try {
    const req = await fetch('/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ document: text, redirect: false })
    })
    const data = await req.json()
    document.querySelector("#results").style = "display: block;"
    document.querySelector("#result").textContent = `${data.classification * 100}%`
    document.querySelector("#confidence").textContent = data.confidence
  } catch(e) {
    console.error(e)
    document.querySelector("#results").style = "display: block;"
    document.querySelector("#result").textContent = "There was an error getting the classification. Try again later."
    document.querySelector("#result").style = "color: red;"
  }
}
const form = document.querySelector("#data-form")
form.addEventListener("submit", handleSubmit)