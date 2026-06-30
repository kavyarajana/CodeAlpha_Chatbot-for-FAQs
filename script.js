// CodeAlpha Task 2: FAQ Chatbot using TF-IDF and Cosine Similarity
const faqData = [
    {
        question: "What is your return policy?",
        answer: "We offer 30-day return policy. Items must be unused with original packaging."
    },
    {
        question: "How long does shipping take?",
        answer: "Standard shipping takes 5-7 business days. Express shipping takes 2-3 days."
    },
    {
        question: "Do you offer international shipping?",
        answer: "Yes, we ship to 50+ countries. International shipping takes 10-15 days."
    },
    {
        question: "How can I track my order?",
        answer: "You can track your order using the tracking ID sent to your email after shipping."
    },
    {
        question: "What payment methods do you accept?",
        answer: "We accept Credit Card, Debit Card, UPI, Net Banking, and Cash on Delivery."
    },
    {
        question: "Can I cancel my order?",
        answer: "Orders can be cancelled within 2 hours of placing. After that contact support."
    },
    {
        question: "Do you have customer support?",
        answer: "Yes, 24/7 customer support via chat, email: support@company.com, phone: 1800-123-456."
    },
    {
        question: "What is the warranty period?",
        answer: "All products have 1 year manufacturer warranty. Extended warranty available."
    },
    {
        question: "How do I reset my password?",
        answer: "Click Forgot Password on login page. Enter email to receive reset link."
    },
    {
        question: "Are products available in stores?",
        answer: "Yes, check our store locator on website for nearest retail store."
    }
];

const stopWords = new Set(['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was',
    'will', 'with', 'what', 'how', 'can', 'i', 'do', 'you', 'my']);

function preprocessText(text) {
    return text.toLowerCase()
       .replace(/[^\w\s]/g, '')
       .split(/\s+/)
       .filter(word =>!stopWords.has(word) && word.length > 2)
       .join(' ');
}

function computeTFIDF(documents) {
    const vocab = new Set();
    const processedDocs = documents.map(doc => {
        const words = preprocessText(doc).split(' ');
        words.forEach(word => vocab.add(word));
        return words;
    });

    const vocabArray = Array.from(vocab);
    const tfidfMatrix = [];

    const idf = {};
    vocabArray.forEach(term => {
        const docCount = processedDocs.filter(doc => doc.includes(term)).length;
        idf[term] = Math.log(documents.length / (docCount + 1));
    });

    processedDocs.forEach(doc => {
        const tf = {};
        doc.forEach(word => {
            tf[word] = (tf[word] || 0) + 1;
        });

        const vector = vocabArray.map(term => {
            const tfValue = (tf[term] || 0) / doc.length;
            return tfValue * idf[term];
        });
        tfidfMatrix.push(vector);
    });

    return { matrix: tfidfMatrix, vocab: vocabArray, idf: idf };
}

function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }

    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

function transformQuery(query, vocab, idf) {
    const words = preprocessText(query).split(' ');
    const tf = {};
    words.forEach(word => {
        tf[word] = (tf[word] || 0) + 1;
    });

    return vocab.map(term => {
        const tfValue = (tf[term] || 0) / words.length;
        return tfValue * (idf[term] || 0);
    });
}

const questions = faqData.map(faq => faq.question);
const { matrix, vocab, idf } = computeTFIDF(questions);

function getChatbotResponse(userQuestion) {
    const userVector = transformQuery(userQuestion, vocab, idf);

    let bestMatch = -1;
    let bestScore = 0;

    matrix.forEach((faqVector, index) => {
        const score = cosineSimilarity(userVector, faqVector);
        if (score > bestScore) {
            bestScore = score;
            bestMatch = index;
        }
    });

    if (bestScore > 0.1 && bestMatch!== -1) {
        return {
            answer: faqData[bestMatch].answer,
            question: faqData[bestMatch].question,
            score: bestScore
        };
    } else {
        return {
            answer: "Sorry, I couldn't find a relevant answer. Please contact support@company.com or call 1800-123-456",
            question: null,
            score: bestScore
        };
    }
}

const chatBox = document.getElementById('chatBox');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const faqList = document.getElementById('faqList');

function addMessage(content, isUser = false, confidence = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = isUser? 'user-message' : 'bot-message';

    let html = `<div class="message-content">${content}`;
    if (confidence!== null &&!isUser) {
        html += `<div class="confidence">Matched: ${confidence.question} | Confidence: ${(confidence.score * 100).toFixed(0)}%</div>`;
    }
    html += `</div>`;

    messageDiv.innerHTML = html;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function handleUserInput() {
    const question = userInput.value.trim();
    if (!question) return;

    addMessage(question, true);
    userInput.value = '';

    setTimeout(() => {
        const response = getChatbotResponse(question);
        const confidence = response.question? { question: response.question, score: response.score } : null;
        addMessage(response.answer, false, confidence);
    }, 500);
}

sendBtn.addEventListener('click', handleUserInput);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleUserInput();
});

faqData.forEach(faq => {
    const li = document.createElement('li');
    li.textContent = faq.question;
    li.addEventListener('click', () => {
        userInput.value = faq.question;
        handleUserInput();
    });
    faqList.appendChild(li);
});