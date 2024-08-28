'use client'

import Image from "next/image";
import styles from "./page.module.css";
import { Box, Button, Stack, TextField } from '@mui/material'

export default function Home() {

  const sendMessage = async () => {
    console.log("We are preparing the request")
    fetch("http://localhost:8080/api/recommendation", {
      method: "POST",
      body: JSON.stringify({
        content: "I would like to find a teacher that is well organized and pushes me"
      }),
      headers: {
        "Content-type": "application/json; charset=UTF-8"
      }
    })
  }

  return (
    <Button onClick={() => sendMessage()}>
      Click Me
    </Button>
  );
}
