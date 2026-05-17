import http from 'http';

const req = http.request({
  hostname: '127.0.0.1',
  port: 3000,
  path: '/api/chat/stream',
  method: 'POST',
  headers: { 'Content-Type': 'application/json' }
}, (res) => {
  console.log("STATUS:", res.statusCode);
  res.on('data', d => console.log(d.toString()));
});
req.write(JSON.stringify({
  message: ['Hi'],
  system_prompt: 'Test',
  use_rag: false,
  temperature: 0.1,
  max_tokens: 50
}));
req.end();
