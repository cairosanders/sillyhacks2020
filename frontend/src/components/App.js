import React, { Component } from 'react';
import { render } from 'react-dom'

class App extends Component {
  constructor(props){
    super(props)
    this.state = {
      data: [],
    };
  }


  componentDidMount(){
    fetch("meme/api/")
      .then(response=>{
        if(response.status > 400) {
          return this.setState(()=>{
            return "something went wrong";
          });
        }
        return response.json();
      })
      .then(data=>{
        this.setState(()=>{
          return{
            data,
            loaded: true
          };
        });
      });
  }

  render(){
    return(
      <ul>
        <h1>Silly Hacks 2020</h1>
        {this.state.data.map(meme=>{
          return(
            <li key={meme.id}>
            {meme.description} - {meme.img_url}
            </li>
          );
        })}
      </ul>
    );
  }
}
