// static/report.js
document.addEventListener('DOMContentLoaded', async function() {
    try {
        // Fetch data from server
        const response = await fetch('/get-report-data');
        if (!response.ok) {
            throw new Error('Failed to fetch report data');
        }
        
        const { predictionData, formData } = await response.json();
        
        if (!predictionData || !formData) {
            window.location.href = '/';
            return;
        }
        
        // Rest of your existing code...
        displayPatientInfo(formData);
        displayRiskSummary(predictionData);
        generateDetailedReport(predictionData, formData);
        
    } catch (error) {
        console.error('Error loading report:', error);
        window.location.href = '/';
    }
    
    // Display patient information
    displayPatientInfo(formData);
    
    // Display risk summary
    displayRiskSummary(predictionData);
    
    // Generate and display detailed report
    generateDetailedReport(predictionData, formData);
    
    // Set up button event listeners
    document.getElementById('printReport').addEventListener('click', function() {
        window.print();
    });
    
    document.getElementById('backToForm').addEventListener('click', function() {
        window.location.href = '/';
    });
});

function displayPatientInfo(formData) {
    const patientDetails = document.getElementById('patientDetails');
    let html = `
        <p><strong>Age:</strong> ${formData.Age || 'Not provided'}</p>
        <p><strong>Gender:</strong> ${formData.Gender || 'Not provided'}</p>
        <p><strong>BMI:</strong> ${formData.BMI || 'Not provided'}</p>
        <p><strong>Blood Pressure:</strong> ${formData.Systolic_BP || '?'}/${formData.Diastolic_BP || '?'} mmHg</p>
    `;
    patientDetails.innerHTML = html;
}

function displayRiskSummary(predictionData) {
    const riskSummary = document.getElementById('riskSummary');
    let html = '<div class="risk-summary-grid">';
    
    for (const [disease, prediction] of Object.entries(predictionData)) {
        const riskClass = prediction.risk_level.toLowerCase();
        html += `
            <div class="recommendation-item">
                <h3>${disease.replace(/_/g, ' ')}</h3>
                <p>Risk Level: <span class="risk-${riskClass}">${prediction.risk_level}</span></p>
                <p>Probability: ${(prediction.probability * 100).toFixed(1)}%</p>
            </div>
        `;
    }
    
    html += '</div>';
    riskSummary.innerHTML = html;
}

async function generateDetailedReport(predictionData, formData) {
    try {
        // Send data to server for report generation
        const response = await fetch('/generate-report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                predictions: predictionData,
                patientData: formData
            }),
        });
        
        if (!response.ok) {
            throw new Error('Failed to generate report');
        }
        
        const report = await response.json();
        
        // Display the generated report sections
        document.getElementById('detailedRecommendations').innerHTML = formatReportSection(report.detailed_recommendations);
        document.getElementById('preventionPlan').innerHTML = formatReportSection(report.prevention_plan);
        document.getElementById('treatmentOptions').innerHTML = formatReportSection(report.treatment_options);
        document.getElementById('lifestyleModifications').innerHTML = formatReportSection(report.lifestyle_modifications);
        document.getElementById('monitoringPlan').innerHTML = formatReportSection(report.monitoring_plan);
        
    } catch (error) {
        console.error('Error generating report:', error);
        document.getElementById('detailedRecommendations').innerHTML = 
            '<p>Error generating detailed recommendations. Please try again.</p>';
    }
}

function formatReportSection(content) {
    if (typeof content === 'string') {
        // Simple string content
        return `<div class="recommendation-item"><p>${content.replace(/\n/g, '<br>')}</p></div>`;
    } else if (Array.isArray(content)) {
        // Array of items
        let html = '';
        content.forEach(item => {
            html += `<div class="recommendation-item"><p>${item.replace(/\n/g, '<br>')}</p></div>`;
        });
        return html;
    } else {
        // Object with sections
        let html = '';
        for (const [section, text] of Object.entries(content)) {
            html += `
                <div class="recommendation-item">
                    <h3>${section.replace(/_/g, ' ')}</h3>
                    <p>${text.replace(/\n/g, '<br>')}</p>
                </div>
            `;
        }
        return html;
    }
}