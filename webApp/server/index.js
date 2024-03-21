const express = require("express");
const axios = require("axios");
const { spawn } = require('child_process');

const PORT = process.env.PORT || 3001;
const app = express();

function runModels(data) {
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python', ['./pythonProcess/main.py', JSON.stringify(data)]);

        let pythonOutput = '';

        pythonProcess.stdout.on('data', (data) => {
            pythonOutput += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            console.error(`Python error: ${data}`);
        });

        pythonProcess.on('close', (code) => {
            if (code === 0) {
                resolve(pythonOutput);
            } else {
                reject(`Python process exited with code ${code}`);
            }
        });
    });
}

app.get("/api", (req, res) => {
    axios.get("https://www.hydroquebec.com/data/documents-donnees/donnees-ouvertes/json/demande.json")
        .then(async (response) => {
            try {
                // Call the Python script with the Axios response data as argument
                const pythonOutput = await runModels(response.data);
                // Send the output of the Python script as the response
                res.send(pythonOutput);
            } catch (error) {
                console.error(error);
                res.status(500).send("Internal Server Error");
            }
        })
        .catch((error) => {
            console.error(error);
            res.status(500).send("Internal Server Error");
        });
});

app.listen(PORT, () => {
    console.log(`Server is listening on port ${PORT}`);
});