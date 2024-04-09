import React from "react";
import "./App.css";
import { LineChart } from "@mui/x-charts";

function App() {
    const [data, setData] = React.useState(null);

    const keyToLabel = {
        real: "Average Power Output (MW)",
        gru: "GRU",
        lstm: "LSTM",
        rnn: "RNN",
        scnn: "SCNN",
        svr: "SVR",
        dnn: "DNN",
    };

    const colors = {
        real: "blue",
        gru: "lightgreen",
        lstm: "orange",
        rnn: "red",
        scnn: "pink",
        svr: "yellow",
        dnn: "darkgrey",
    };

    const customize = {
        height: 300,
        legend: { hidden: true },
        margin: { top: 5 },
    };

    React.useEffect(() => {
        fetch("/api")
            .then((res) => res.json())
            .then((data) => setData(data))
            .then((data) => console.log(data));
    }, []);

    return (
        <div className="App">
            <header className="App-header">
                {!data ? (
                    <p>Loading...</p>
                ) : (
                    <LineChart
                        xAxis={[
                            {
                                dataKey: "Date/Time",
                                valueFormatter: (value) => value.toString(),
                            },
                        ]}
                        series={Object.keys(keyToLabel).map((key) => ({
                            dataKey: key,
                            label: keyToLabel[key],
                            color: colors[key],
                            showMark: false,
                        }))}
                        dataset={data.data}
                        {...customize}
                    />
                )}
            </header>
        </div>
    );
}

export default App;
