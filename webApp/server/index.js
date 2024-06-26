const express = require("express");
const axios = require("axios");
const { spawn } = require("child_process");
const fs = require("fs").promises;
const fsNow = require("fs");

const PORT = process.env.PORT || 3001;
const app = express();

app.get("/api", async (req, res) => {
    try {
        // Fetch data from Hydro-Québec API
        const hydroResponse = await axios.get(
            "https://www.hydroquebec.com/data/documents-donnees/donnees-ouvertes/json/demande.json"
        );
        const hydroData = hydroResponse.data;

        const start = hydroData?.dateStart.split("T")[0];
        const end = hydroData?.recentHour.split("T")[0];

        const weather = await Promise.all([
            {
                city: "MTL",
                wData: await axios.get(
                    `https://api.open-meteo.com/v1/forecast?latitude=45.28&longitude=-73.44&minutely_15=temperature_2m&start_date=${start}&end_date=${end}`
                ),
            },
            {
                city: "QC",
                wData: await axios.get(
                    `https://api.open-meteo.com/v1/forecast?latitude=46.47&longitude=-71.23&minutely_15=temperature_2m&start_date=${start}&end_date=${end}`
                ),
            },
            {
                city: "SHE",
                wData: await axios.get(
                    `https://api.open-meteo.com/v1/forecast?latitude=45.26&longitude=-71.41&minutely_15=temperature_2m&start_date=${start}&end_date=${end}`
                ),
            },
            {
                city: "GAT",
                wData: await axios.get(
                    `https://api.open-meteo.com/v1/forecast?latitude=45.31&longitude=-75.33&minutely_15=temperature_2m&start_date=${start}&end_date=${end}`
                ),
            },
        ]);

        const weatherData = {};

        weather.forEach(({ city, wData }) => {
            weatherData[city] = wData?.data?.minutely_15;
        });

        const rearrangedData = {};
        Object.entries(weatherData).forEach(([city, cityData]) => {
            cityData.time.forEach((timestamp, index) => {
                if (!rearrangedData[timestamp]) {
                    rearrangedData[timestamp] = {};
                }
                rearrangedData[timestamp][city] =
                    cityData.temperature_2m[index];
            });
        });

        const wet = "./data/wet.json";
        const pow = "./data/pow.json";
        let out = "./data/out.json";
        let pythonOutput;

        await fs.writeFile(wet, JSON.stringify(rearrangedData));
        await fs.writeFile(pow, JSON.stringify(hydroData));

        const pythonProcess = spawn("python", [
            "./server/pythonProcess/main.py",
            pow,
            wet,
        ]);

        pythonProcess.stderr.on("data", (data) => {
            console.error(`Python error: ${data}`);
        });

        pythonProcess.on("close", (code) => {
            if (code == 0) {
                //read from out.json
                fsNow.readFile(out, "utf8", (err, data) => {
                    if (err) {
                        console.error(err);
                        return;
                    }
                    pythonOutput = JSON.parse(data);

                    //delete files
                    fs.unlink(wet);
                    fs.unlink(pow);
                    fs.unlink(out);

                    res.send(pythonOutput);
                });
            }
        });
    } catch (error) {
        console.error(error);
        res.status(500).send("Internal Server Error");
    }
});

app.listen(PORT, () => {
    console.log(`Server is listening on port ${PORT}`);
});
