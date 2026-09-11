document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById('audio-file');
    const submitBtn = document.getElementById('submit-btn');
    const loader = document.getElementById('loader');
    const resultSection = document.getElementById('result-section');
    const riskTitleSpan = document.querySelector('#risk-title span');
    
    if (fileInput.files.length === 0) return;

    submitBtn.disabled = true;
    submitBtn.innerText = "Processing...";
    loader.style.display = 'block';
    resultSection.style.display = 'none';

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const response = await fetch("http://127.0.0.1:8000/api/analyze-audio", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();

        // Inject data
        riskTitleSpan.innerText = data.risk_level.toUpperCase();
        document.getElementById('score-text').innerText = data.ml_score_percent;
        document.getElementById('duration-text').innerText = data.duration_seconds;
        document.getElementById('summary-text').innerText = data.summary;
        document.getElementById('action-text').innerText = data.recommendation;

        // Dynamic Styling via JS
        if (data.risk_level.includes("High")) {
            resultSection.style.borderLeft = "6px solid #ef4444"; // Red
            riskTitleSpan.style.color = "#ef4444";
        } else if (data.risk_level.includes("Medium")) {
            resultSection.style.borderLeft = "6px solid #f59e0b"; // Yellow
            riskTitleSpan.style.color = "#f59e0b";
        } else {
            resultSection.style.borderLeft = "6px solid #10b981"; // Green
            riskTitleSpan.style.color = "#10b981";
        }

        resultSection.style.display = 'block';

    } catch (error) {
        alert("Failed to connect to backend.");
        console.error(error);
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerText = "Analyze Audio Engine";
        loader.style.display = 'none';
    }
});