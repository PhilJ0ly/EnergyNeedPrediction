import React from "react";
import "./App.css";
import { LineChart } from "@mui/x-charts";

function App() {
    const [data, setData] = React.useState(null);
    const apiCalled = React.useRef(false);

    React.useEffect(() => {
        if(!apiCalled.current){
            fetch("/api")
            .then((res) => res.json())
            .then((data) => {
                apiCalled.current = true;
                setData(data);
            })
        }
    }, [apiCalled.current]);

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
        legend: { hidden: false },
        margin: { top: 5 },
    };

    const formatDateTime = (timestamp) => {
        const date = new Date(timestamp);
        return date.toLocaleString(); // Adjust format as needed
    };

    return (
        <div className="App">
            <header className="App-header">
                <h1>Power Consumption Predictions for Québec</h1>
                {/* {data ? ({
                    startDate = new Date(data.data[0]["Date/Time"]);
                    endDate = new Date(data.data[0]["Date/Time"]);

                }
                    <h2>From {new Date(data.data[0]["Date/Time"])}
                    </h2>
                ):(<h2>2 Day Interval</h2>)} */}

                {!data ? (
                    <p>Loading...</p>
                ) : (
                    <>
                    <h4>From {formatDateTime(data.data[0]["Date/Time"])} to {formatDateTime(data.data[data.data.length-1]["Date/Time"])}</h4>
                    <LineChart
                        xAxis={[
                            {
                                dataKey: "Date/Time",
                                valueFormatter: (value) => {
                                    const date = new Date(value);
                                    return date.toLocaleTimeString();
                                },
                            },
                        ]}
                        series={Object.keys(keyToLabel).map((key) => ({
                            dataKey: keyToLabel[key],
                            label: keyToLabel[key],
                            color: colors[key],
                            showMark: false,
                        }))}
                        dataset={data.data}
                        {...customize}
                    />
                    </>
                )}
            </header>
        </div>
    );
}

export default App;
