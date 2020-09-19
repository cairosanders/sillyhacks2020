import React, { Component } from "react";
import Game from "./Game";
import "./App.css";

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      data: [],
    };
  }

  componentDidMount() {
    fetch("meme/api/")
      .then((response) => {
        if (response.status > 400) {
          return this.setState(() => {
            return "something went wrong";
          });
        }
        return response.json();
      })
      .then((data) => {
        this.setState(() => {
          return {
            data,
            loaded: true,
          };
        });
      });
  }

  render() {
    return (
      <div className="App">
        <div class="heading">
          <h1>Silly Hacks 2020</h1>
        </div>

        {this.state.data.map((meme) => {
          return (
            <li key={meme.id}>
              {meme.description} - {meme.img_url}
            </li>
          );
        })}

        <Game />
      </div>
    );
  }
}

export default App;
