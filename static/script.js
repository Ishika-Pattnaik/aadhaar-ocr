const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('drop-zone').querySelector('input');
const loading = document.getElementById('loading');
const resultSection = document.getElementById('result-section');
const errorMessage = document.getElementById('error-message');
const previewImage = document.getElementById('preview-image');
const resultName = document.getElementById('extracted-name');
const resultAadhaar = document.getElementById('extracted-aadhaar');
const resultConf = document.getElementById('confidence-score');
const checksumStatus = document.getElementById('checksum-status');
const rawJson = document.getElementById('raw-json');

// Drag & Drop Events
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

async function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) {
        showError('Please upload a valid image file (JPG, PNG).');
        return;
    }

    // Reset UI
    showError(null);
    dropZone.style.display = 'none';
    loading.classList.remove('hidden');
    resultSection.classList.add('hidden');

    // Show Preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
    };
    reader.readAsDataURL(file);

    // Prepare upload
    const formData = new FormData();
    formData.append('file', file);

    // API Key (from server.py default)
    const apiKey = 'your-secret-api-key-change-in-production';

    try {
        const response = await fetch('/extract', {
            method: 'POST',
            headers: {
                'X-API-Key': apiKey
            },
            body: formData
        });

        const data = await response.json();

        if (response.ok && data.success) {
            displayResults(data.data);
        } else {
            throw new Error(data.message || 'Extraction failed');
        }
    } catch (err) {
        console.error(err);
        showError(err.message || 'An error occurred while processing the image.');
        // Reset to upload state after error
        loading.classList.add('hidden');
        dropZone.style.display = 'block';
    } finally {
        loading.classList.add('hidden');
    }
}

function displayResults(data) {
    loading.classList.add('hidden');
    resultSection.classList.remove('hidden');

    // Fill data
    resultName.textContent = data.extracted_name || 'Not Found';
    resultAadhaar.textContent = data.extracted_aadhaar || 'Not Found';
    resultConf.textContent = (data.confidence_score * 100).toFixed(1) + '%';

    // Checksum Status
    if (data.aadhaar_checksum_valid) {
        checksumStatus.className = 'indicator valid';
        checksumStatus.innerHTML = '<span class="dot"></span> Checksum Valid';
    } else {
        checksumStatus.className = 'indicator invalid';
        checksumStatus.innerHTML = '<span class="dot"></span> Checksum Invalid';
    }

    // Raw JSON
    rawJson.textContent = JSON.stringify(data, null, 2);
}

function toggleJson() {
    rawJson.classList.toggle('hidden');
}

function resetUI() {
    showError(null);
    dropZone.style.display = 'block';
    loading.classList.add('hidden');
    resultSection.classList.add('hidden');
    fileInput.value = '';
    previewImage.src = '';
}

function showError(msg) {
    const errorText = document.getElementById('error-text');
    if (msg) {
        errorText.textContent = msg;
        errorMessage.classList.remove('hidden');
    } else {
        errorMessage.classList.add('hidden');
    }
}
