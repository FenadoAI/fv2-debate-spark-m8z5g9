import { useState } from "react";
import axios from "axios";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8001';
const API = `${API_BASE}/api`;

const DebatePrep = () => {
  const [topic, setTopic] = useState("");
  const [loading, setLoading] = useState(false);
  const [debateData, setDebateData] = useState(null);
  const [error, setError] = useState(null);

  const handleGenerate = async () => {
    if (!topic.trim()) {
      setError("Please enter a topic");
      return;
    }

    setLoading(true);
    setError(null);
    setDebateData(null);

    try {
      const response = await axios.post(`${API}/debate`, {
        topic: topic.trim()
      });

      if (response.data.success) {
        setDebateData(response.data);
      } else {
        setError(response.data.error || "Failed to generate arguments");
      }
    } catch (err) {
      setError("Network error: Unable to connect to the server");
      console.error("Error generating debate arguments:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Debate Prep Generator
          </h1>
          <p className="text-gray-600 text-lg">
            Get balanced pro and con arguments for any debate topic
          </p>
        </div>

        <div className="max-w-2xl mx-auto mb-8">
          <Card>
            <CardHeader>
              <CardTitle>Enter Your Debate Topic</CardTitle>
              <CardDescription>
                Type any topic you'd like to explore both sides of
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex gap-3">
                <Input
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  placeholder="e.g., Should AI be used in education?"
                  className="flex-1"
                  onKeyPress={(e) => {
                    if (e.key === "Enter") {
                      handleGenerate();
                    }
                  }}
                />
                <Button
                  onClick={handleGenerate}
                  disabled={loading}
                  className="px-6"
                >
                  {loading ? "Generating..." : "Generate"}
                </Button>
              </div>
              {error && (
                <div className="mt-4 p-3 bg-red-100 border border-red-200 rounded-md">
                  <p className="text-red-800 text-sm">{error}</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {debateData && (
          <div className="grid md:grid-cols-2 gap-6">
            {/* Pro Arguments */}
            <Card className="border-green-200 bg-green-50">
              <CardHeader>
                <CardTitle className="text-green-800 text-xl">
                  Arguments For
                </CardTitle>
                <CardDescription className="text-green-700">
                  Supporting the position: "{debateData.topic}"
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {debateData.arguments_for.map((argument, index) => (
                  <div key={index} className="bg-white p-4 rounded-lg border border-green-200">
                    <h4 className="font-semibold text-green-900 mb-2">
                      {argument.point}
                    </h4>
                    <ul className="space-y-1">
                      {argument.supporting_facts.map((fact, factIndex) => (
                        <li key={factIndex} className="text-sm text-green-700 flex items-start">
                          <span className="mr-2 text-green-500">•</span>
                          {fact}
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </CardContent>
            </Card>

            {/* Con Arguments */}
            <Card className="border-red-200 bg-red-50">
              <CardHeader>
                <CardTitle className="text-red-800 text-xl">
                  Arguments Against
                </CardTitle>
                <CardDescription className="text-red-700">
                  Opposing the position: "{debateData.topic}"
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {debateData.arguments_against.map((argument, index) => (
                  <div key={index} className="bg-white p-4 rounded-lg border border-red-200">
                    <h4 className="font-semibold text-red-900 mb-2">
                      {argument.point}
                    </h4>
                    <ul className="space-y-1">
                      {argument.supporting_facts.map((fact, factIndex) => (
                        <li key={factIndex} className="text-sm text-red-700 flex items-start">
                          <span className="mr-2 text-red-500">•</span>
                          {fact}
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        )}

        {loading && (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <p className="mt-4 text-gray-600">Generating balanced arguments...</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default DebatePrep;