const express = require("express");
const axios = require("axios");

const PORT = process.env.PORT || 3001;

const app = express();

app.get("/api", (req, res) => {
    axios({
        method: "get",
        url: "https://www.hydroquebec.com/data/documents-donnees/donnees-ouvertes/json/demande.json",
    })
        .then((response) => {
            res.send(response.data);
        })
        .catch((error) => {
            console.log(error);
            res.status(500).send("Internal Server Error");
        });
});

app.listen(PORT, () => {
    console.log(`Server is listening on port ${PORT}`);
});
