const words = ['HANGMAN', 'JAVASCRIPT', 'COMPUTER', 'PROGRAMMING', 'DEVELOPER'];
let chosenWord = '';
let guessedLetters = [];
let attemptsLeft = 6;

function startGame() {
  attemptsLeft = 6;
  guessedLetters = [];
  chosenWord = words[Math.floor(Math.random() * words.length)];
  
  let display = '';
  for (let i = 0; i < chosenWord.length; i++) {
    display += '_ ';
  }
  
  document.getElementById('word-display').innerText = display;
  document.getElementById('attempt-count').innerText = attemptsLeft;
  document.getElementById('hangman-img').src = '../static/images/hangman-0.png';
}

function makeGuess() {
  if (attemptsLeft === 0) {
    alert('Out of attempts! The word was: ' + chosenWord);
    resetGame();
    return;
  }

  const guessInput = document.getElementById('guess-input');
  const guess = guessInput.value.toUpperCase();

  if (!/[A-Z]/.test(guess) || guessedLetters.includes(guess)) {
    alert('Please enter a valid letter that you haven\'t guessed before.');
    return;
  }

  guessedLetters.push(guess);
  guessInput.value = '';

  let display = '';
  let correctGuess = false;
  
  for (const letter of chosenWord) {
    if (guessedLetters.includes(letter)) {
      display += letter + ' ';
    } else {
      display += '_ ';
    }
    if (letter === guess) {
      correctGuess = true;
    }
  }

  document.getElementById('word-display').innerText = display;
  document.getElementById('guesses').innerText = `Guessed letters: ${guessedLetters.join(', ')}`;

  if (!display.includes('_')) {
    alert('You won! The word was: ' + chosenWord);
    resetGame();
  } else if (!correctGuess) {
    attemptsLeft--;

    document.getElementById('attempt-count').innerText = attemptsLeft;
    document.getElementById('img').src = `../static/images/${6 - attemptsLeft}.png`;
    if (attemptsLeft === 0) {
      setTimeout(function() {
        alert('Out of attempts! The word was: ' + chosenWord);
        resetGame();
      }, 1000);


    }
  }
}

function resetGame() {
  document.getElementById('word-display').innerText = '';
  document.getElementById('guesses').innerText = '';
  document.getElementById('attempt-count').innerText = attemptsLeft;
  document.getElementById('img').src = '../static/images/0.png';
  startGame();
}

startGame();