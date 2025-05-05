import logo from './logo.svg';
import './App.css';

import axios from "axios";
import { useEffect, useState } from 'react';

function App() {
  const [buh, setBuh] = useState({});
  useEffect(() => {

    const hello = async () => {
      const response = await axios.get("http://localhost:5050/shape/");
      const data = await response.data;
      console.log(data)
    }

    hello();
    
  },[]);

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

export default App;
