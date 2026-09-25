function startWebcam() {

    const webcam =
        document.getElementById(
            "webcamStream"
        );

    webcam.src =
        "/webcam-feed?time="
        + new Date().getTime();
}



function stopWebcam() {

    const webcam =
        document.getElementById(
            "webcamStream"
        );

    webcam.src = "";
}



function refreshPage() {

    window.location.reload();
}