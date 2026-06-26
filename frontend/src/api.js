const API_BASE_URL = "http://127.0.0.1:8000";

async function parseResponse(response) {
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.detail || "Request failed");
  }

  return data;
}

export async function getDocuments() {
  const response = await fetch(`${API_BASE_URL}/documents`);
  return parseResponse(response);
}

export async function indexDocument(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/documents/index`, {
    method: "POST",
    body: formData,
  });

  return parseResponse(response);
}

export async function askQuestion(question, topK = 4) {
  const response = await fetch(`${API_BASE_URL}/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      top_k: topK,
    }),
  });

  return parseResponse(response);
}
