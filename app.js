const express = require('express');
const { Storage } = require('@google-cloud/storage');
const tf = require('@tensorflow/tfjs-node');
const csv = require('csv-parser');
const natural = require('natural');
const stream = require('stream');

const app = express();
app.use(express.json());

// Initialize storage with authentication
const storage = new Storage({
  keyFilename: './lokal-ind-04e68df8c563.json'
});
const bucketName = 'lokalind-nextword';

let model;
let tokenizer;
let df_items = [];
const max_sequence_len = 10;

async function loadModelAndData() {
  // Load model from private Cloud Storage
  const modelFile = storage.bucket(bucketName).file('model_predict_next_words.h5');
  const [modelBuffer] = await modelFile.download();
  model = await tf.loadLayersModel(tf.io.fileSystem(modelBuffer));

  // Load CSV from private Cloud Storage
  const csvFile = storage.bucket(bucketName).file('API_DS.csv');
  
  return new Promise((resolve, reject) => {
    const buffers = [];
    csvFile.createReadStream()
      .on('data', (chunk) => buffers.push(chunk))
      .on('end', () => {
        const csvContent = Buffer.concat(buffers).toString();
        const csvStream = new stream.Readable();
        csvStream.push(csvContent);
        csvStream.push(null);

        csvStream
          .pipe(csv())
          .on('data', (row) => df_items.push(row.Fixedd))
          .on('end', () => {
            tokenizer = new natural.WordTokenizer();
            const allWords = df_items.flatMap(item => tokenizer.tokenize(item));
            tokenizer.word_index = {};
            allWords.forEach((word, index) => {
              if (!tokenizer.word_index[word]) {
                tokenizer.word_index[word] = index + 1;
              }
            });
            resolve();
          });
      })
      .on('error', reject);
  });
}

function texts_to_sequences(texts) {
    return texts.map(text => 
      tokenizer.tokenize(text).map(word => tokenizer.word_index[word] || 0)
    );
  }

function pad_sequences(sequences, maxlen, padding = 'pre') {
    return sequences.map(seq => {
      if (seq.length > maxlen) {
        return seq.slice(seq.length - maxlen);
      }
      const pad = Array(Math.max(0, maxlen - seq.length)).fill(0);
      return padding === 'pre' ? [...pad, ...seq] : [...seq, ...pad];
    });
  }
  
  async function make_prediction(seed_text, next_words = 1) {
    let output_text = seed_text;
    for (let i = 0; i < next_words; i++) {
      const token_list = texts_to_sequences([seed_text])[0];
      const padded_sequence = pad_sequences([token_list], max_sequence_len - 1, 'pre');
      const predicted = tf.argMax(model.predict(tf.tensor2d(padded_sequence)), -1).dataSync()[0];
      let output_word = "";
      for (const [word, index] of Object.entries(tokenizer.word_index)) {
        if (index === predicted) {
          output_word = word;
          break;
        }
      }
      seed_text += " " + output_word;
      output_text += " " + output_word;
    }
    return output_text;
  }
  
  app.post('/predict', async (req, res) => {
    const { seed_text, next_words = 1 } = req.body;
    try {
      const predicted_text = await make_prediction(seed_text, next_words);
      res.json({ predicted_text });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });
  
  const PORT = process.env.PORT || 8080;
  app.listen(PORT, async () => {
    await loadModelAndData();
    console.log(`Server running on port ${PORT}`);
  });