import { NextResponse } from 'next/server'
import { Pinecone } from '@pinecone-database/pinecone'
import { pipeline } from '@xenova/transformers';

const systemPrompt = `
You are a rate my professor agent to help students find classes, that takes in user questions and answers them.
For every user question, the top 3 professors that match the user question are returned.
Use them to answer the question if needed.
`

export async function POST(request: Request) {
    const data = await request.json()

    let classifier = await pipeline('feature-extraction');
    // We'll add more code here in the following steps
}