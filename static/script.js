document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('patientForm');
    const resultsDiv = document.getElementById('results');

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        const jsonData = {};
        formData.forEach((value, key) => {
            // Convert numeric strings to numbers
            jsonData[key] = isNaN(value) ? value : Number(value);
        });

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(jsonData),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            let resultsHtml = '<h2>Prediction Results:</h2>';
            for (const [disease, prediction] of Object.entries(data)) {
                resultsHtml += `
                    <div class="prediction-result">
                        <h3>${disease}</h3>
                        <p>Probability: ${prediction.probability.toFixed(2)}</p>
                        <p>Risk Level: ${prediction.risk_level}</p>
                    </div>
                `;
            }
            resultsDiv.innerHTML = resultsHtml;
            resultsDiv.style.display = 'block';
        })
        .catch((error) => {
            console.error('Error:', error);
            resultsDiv.innerHTML = '<p>An error occurred while processing your request. Please try again.</p>';
            resultsDiv.style.display = 'block';
        })

        .then(data => {
            let resultsHtml = '<h2>Prediction Results:</h2>';
            for (const [disease, prediction] of Object.entries(data)) {
                resultsHtml += `
                    <div class="prediction-result">
                        <h3>${disease}</h3>
                        <p>Probability: ${prediction.probability.toFixed(2)}</p>
                        <p>Risk Level: ${prediction.risk_level}</p>
                    </div>
                `;
            }
            resultsDiv.innerHTML = resultsHtml;
            resultsDiv.style.display = 'block';

            // Show the report link
            document.getElementById('reportLink').style.display = 'block';
        });
    });

    // Optional: Add form validation
    function validateForm() {
        let isValid = true;
        const inputs = form.querySelectorAll('input[required], select[required]');
        inputs.forEach(input => {
            if (!input.value.trim()) {
                isValid = false;
                input.classList.add('error');
            } else {
                input.classList.remove('error');
            }
        });
        return isValid;
    }

    // Optional: Add real-time BMI calculation
    const heightInput = document.getElementById('height');
    const weightInput = document.getElementById('weight');
    const bmiInput = document.getElementById('bmi');

    function calculateBMI() {
        const height = heightInput.value / 100; // convert cm to m
        const weight = weightInput.value;
        if (height && weight) {
            const bmi = (weight / (height * height)).toFixed(2);
            bmiInput.value = bmi;
        }
    }

    heightInput.addEventListener('input', calculateBMI);
    weightInput.addEventListener('input', calculateBMI);
});