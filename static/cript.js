function updatePersonCount(camId) {
    fetch(`/person_count/${camId}`)
        .then(res => res.json())
        .then(data => {
            const el = document.getElementById(`count-${camId}`);
            if (el) el.textContent = data.count;
        })
        .catch(console.error);
}

function startUpdatingCounts(cameraIds) {
    cameraIds.forEach(id => {
        updatePersonCount(id);
        setInterval(() => updatePersonCount(id), 2000);
    });
}

window.onload = function() {
    const cams = Array.from(document.querySelectorAll('.camera-container'))
                    .map(div => div.getAttribute('data-cam-id'));
    startUpdatingCounts(cams);
};
