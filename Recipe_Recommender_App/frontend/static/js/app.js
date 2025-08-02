document.getElementById('uploadForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const fileInput = document.getElementById('imageInput');
    const description = document.getElementById('descriptionInput').value;
    const file = fileInput.files[0];
    if (!file) {
        alert("Please select an image.");
        return;
    }

    const formData = new FormData();
    
    formData.append("file", file);
    formData.append('description', description);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        // Update table
        const tbody = document.querySelector('#resultsTable tbody');
        tbody.innerHTML = "";  // Clear existing

        data.forEach(item => {
            const row = `<tr>
                <td>${item.recipe_id}</td>
                <td>${item.name}</td>
                <td>${item.rating}</td>
            </tr>`;
            tbody.innerHTML += row;
        });

    } catch (error) {
        console.error("Error:", error);
        alert("Something went wrong!");
    }
});
