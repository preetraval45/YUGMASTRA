import { NextRequest, NextResponse } from 'next/server';

const SIMULATION_API_URL = process.env.SIMULATION_API_URL || 'http://ai-engine:8002';

export async function GET(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  try {
    const path = params.path.join('/');
    const searchParams = request.nextUrl.searchParams.toString();
    const url = `${SIMULATION_API_URL}/api/simulation/${path}${searchParams ? `?${searchParams}` : ''}`;

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error('Simulation API Error:', error);
    return NextResponse.json(
      { error: 'Failed to connect to simulation engine' },
      { status: 500 }
    );
  }
}

export async function POST(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  try {
    const path = params.path.join('/');
    const body = await request.json();
    const url = `${SIMULATION_API_URL}/api/simulation/${path}`;

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error('Simulation API Error:', error);
    return NextResponse.json(
      { error: 'Failed to connect to simulation engine' },
      { status: 500 }
    );
  }
}
