document.addEventListener('DOMContentLoaded', () => {
    // Current Date display
    const dateEl = document.getElementById('current-date');
    if (dateEl) {
        const options = { year: 'numeric', month: 'long', day: 'numeric' };
        dateEl.textContent = new Date().toLocaleDateString('en-US', options);
    }

    // Tab Navigation
    const navItems = document.querySelectorAll('.nav-item');
    const tabContents = document.querySelectorAll('.tab-content');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            // Remove active classes
            navItems.forEach(i => i.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Add active class to clicked button
            item.classList.add('active');

            // Show corresponding tab content
            const tabId = item.getAttribute('data-tab');
            const targetTab = document.getElementById(tabId);
            if (targetTab) {
                targetTab.classList.add('active');
            }
        });
    });

    // Credit Card Utilization Slider Label update
    const ccUtilInput = document.getElementById('Credit_Card_Utilization');
    const ccUtilVal = document.getElementById('cc-util-val');
    
    if (ccUtilInput && ccUtilVal) {
        ccUtilInput.addEventListener('input', (e) => {
            const percent = Math.round(e.target.value * 100);
            ccUtilVal.textContent = `${percent}%`;
        });
    }

    // Form Submission and prediction query
    const form = document.getElementById('credit-form');
    const resultPlaceholder = document.querySelector('.result-placeholder');
    const realResults = document.getElementById('real-results');
    
    // Result fields
    const resScore = document.getElementById('res-score');
    const scoreGauge = document.getElementById('score-gauge');
    const resDecision = document.getElementById('res-decision');
    const resCategory = document.getElementById('res-category');
    const resRiskDesc = document.getElementById('res-risk-desc');
    const resInstallment = document.getElementById('res-installment');
    const resDti = document.getElementById('res-dti');
    const posFactorsList = document.getElementById('pos-factors-list');
    const negFactorsList = document.getElementById('neg-factors-list');

    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            // Show loading animation on button
            const submitBtn = form.querySelector('button[type="submit"]');
            const originalBtnHtml = submitBtn.innerHTML;
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Processing Assessment...';
            
            // Collect form data
            const formData = new FormData(form);
            const payload = {};
            
            // Text/Number fields
            formData.forEach((value, key) => {
                if (key === 'Owns_Home') {
                    payload[key] = 1;
                } else if (key === 'Education_Level') {
                    payload[key] = value;
                } else if (['Age', 'Existing_Loans_Count', 'Loan_Duration_Months', 'Payment_History_Delay'].includes(key)) {
                    payload[key] = parseInt(value, 10);
                } else {
                    payload[key] = parseFloat(value);
                }
            });
            
            // Handle unchecked checkbox for Owns_Home
            if (!formData.has('Owns_Home')) {
                payload['Owns_Home'] = 0;
            }

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(payload),
                });
                
                if (!response.ok) {
                    throw new Error('Server returned an error');
                }
                
                const results = await response.json();
                
                // Hide placeholder and reveal results
                if (resultPlaceholder) resultPlaceholder.classList.add('hidden');
                if (realResults) realResults.classList.remove('hidden');
                
                // Update credit score
                const score = results.credit_score;
                if (resScore) resScore.textContent = score;
                
                // Update SVG gauge
                // Circumference is 2 * pi * r = 2 * 3.14159 * 42 = 263.89 => let's use 263
                const circumference = 263.89;
                const minScore = 300;
                const maxScore = 850;
                const scorePercentage = (score - minScore) / (maxScore - minScore);
                const offset = circumference - (scorePercentage * circumference);
                
                if (scoreGauge) {
                    scoreGauge.style.strokeDasharray = `${circumference}`;
                    scoreGauge.style.strokeDashoffset = `${offset}`;
                }
                
                // Reset and apply colors classes
                const colorClass = results.color_class;
                const coloredElements = [scoreGauge, resDecision, resCategory];
                
                coloredElements.forEach(el => {
                    if (el) {
                        // Remove older status classes
                        el.className.baseVal ? el.className.baseVal = 'gauge-fill' : el.className = '';
                        el.classList.add(colorClass);
                    }
                });
                
                // Update textual output
                if (resDecision) resDecision.textContent = results.decision;
                if (resCategory) resCategory.textContent = results.category;
                if (resRiskDesc) resRiskDesc.textContent = results.risk_description;
                if (resInstallment) resInstallment.textContent = `$${results.monthly_payment}`;
                if (resDti) resDti.textContent = `${(results.debt_to_income * 100).toFixed(1)}%`;
                
                // Update lists
                if (posFactorsList) {
                    posFactorsList.innerHTML = '';
                    results.factors_positive.forEach(f => {
                        const li = document.createElement('li');
                        li.textContent = f;
                        posFactorsList.appendChild(li);
                    });
                }
                
                if (negFactorsList) {
                    negFactorsList.innerHTML = '';
                    results.factors_negative.forEach(f => {
                        const li = document.createElement('li');
                        li.textContent = f;
                        negFactorsList.appendChild(li);
                    });
                }
                
                // Smooth scroll to results on mobile devices
                if (window.innerWidth <= 1024) {
                    const resultCard = document.getElementById('result-panel');
                    if (resultCard) {
                        resultCard.scrollIntoView({ behavior: 'smooth' });
                    }
                }
                
            } catch (error) {
                console.error('Prediction failed:', error);
                alert('Error running risk diagnostics. Please check application settings or python logs.');
            } finally {
                // Restore button state
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalBtnHtml;
            }
        });
    }
});
