import React, { Component } from "react";
import "./App.css";
import Button from "@material-ui/core/Button";
import TextField from "@material-ui/core/TextField";

class Game extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      caption1: "",
      caption2: "",
      caption3: "",
      caption4: "",
      caption5: "",
    };

    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  componentDidMount() {}

  componentDidUpdate() {}

  handleChange(event) {
    this.setState({ [event.target.name]: event.target.value });
    // this.setState({ caption1: event.target.value });
  }

  handleSubmit(event) {
    alert(
      "Submitted captions were: " + this.state.caption1 + this.state.caption2
    );
    event.preventDefault();
  }

  render() {
    return (
      <div className="game">
        <form onSubmit={this.handleSubmit}>
          <TextField
            id="textbox"
            name="caption1"
            label="Caption 1"
            variant="outlined"
            onChange={this.handleChange}
          />

          <TextField
            id="textbox"
            name="caption2"
            label="Caption 2"
            variant="outlined"
            onChange={this.handleChange}
          />

          <TextField
            id="textbox"
            name="caption3"
            label="Caption 3"
            variant="outlined"
            onChange={this.handleChange}
          />

          <TextField
            id="textbox"
            name="caption4"
            label="Caption 4"
            variant="outlined"
            onChange={this.handleChange}
          />

          <TextField
            id="textbox"
            name="caption5"
            label="Caption 5"
            variant="outlined"
            onChange={this.handleChange}
          />

          <Button id="submit" variant="contained" color="primary" type="submit">
            Submit
          </Button>
        </form>
      </div>
    );
  }
}

export default Game;
