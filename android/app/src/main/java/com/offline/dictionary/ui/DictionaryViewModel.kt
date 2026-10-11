package com.offline.dictionary.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.offline.dictionary.db.DictionaryDbHelper
import com.offline.dictionary.db.LookupResult
import com.offline.dictionary.nlp.TermExtractor
import com.offline.dictionary.voice.OfflineVoiceRecognizer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class DictionaryUiState(
    val query: String = "",
    val isSearching: Boolean = false,
    val isListening: Boolean = false,
    val isSpeaking: Boolean = false,
    val currentSpeakingId: String? = null,
    val autoSpeakVoiceResults: Boolean = true,
    val extractedTerm: String? = null,
    val lookupResult: LookupResult? = null,
    val extractionTimeMs: Long = 0L,
    val totalTimeMs: Long = 0L,
    val errorMessage: String? = null
)

class DictionaryViewModel(application: Application) : AndroidViewModel(application) {

    private val extractor by lazy { TermExtractor(application) }
    private val dbHelper by lazy { DictionaryDbHelper(application) }
    val voiceRecognizer by lazy { OfflineVoiceRecognizer(application) }
    val textToSpeech by lazy { com.offline.dictionary.voice.OfflineTextToSpeech(application) }

    private val _uiState = MutableStateFlow(DictionaryUiState())
    val uiState: StateFlow<DictionaryUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch {
            textToSpeech.isSpeaking.collect { speaking ->
                _uiState.value = _uiState.value.copy(isSpeaking = speaking)
            }
        }
        viewModelScope.launch {
            textToSpeech.currentUtteranceId.collect { id ->
                _uiState.value = _uiState.value.copy(currentSpeakingId = id)
            }
        }
    }

    fun onQueryChanged(newQuery: String) {
        _uiState.value = _uiState.value.copy(query = newQuery, errorMessage = null)
    }

    fun toggleAutoSpeakVoiceResults() {
        val next = !_uiState.value.autoSpeakVoiceResults
        if (!next) {
            textToSpeech.stop()
        }
        _uiState.value = _uiState.value.copy(autoSpeakVoiceResults = next)
    }

    fun speakDefinition(headword: String, pos: String, definition: String, example: String?, utteranceId: String) {
        if (_uiState.value.isSpeaking && _uiState.value.currentSpeakingId == utteranceId) {
            textToSpeech.stop()
        } else {
            textToSpeech.speakDefinition(headword, pos, definition, example, utteranceId)
        }
    }

    fun stopSpeaking() {
        textToSpeech.stop()
    }

    fun performSearch(queryToSearch: String = _uiState.value.query, isVoiceQuery: Boolean = false) {
        val q = queryToSearch.trim()
        if (q.isEmpty()) return

        textToSpeech.stop()
        _uiState.value = _uiState.value.copy(isSearching = true, query = q, errorMessage = null)

        viewModelScope.launch {
            val tStart = System.currentTimeMillis()

            val (term, tExtract) = withContext(Dispatchers.Default) {
                val t0 = System.currentTimeMillis()
                val extracted = extractor.extract(q)
                val dt = System.currentTimeMillis() - t0
                Pair(extracted, dt)
            }

            val lookupRes = withContext(Dispatchers.IO) {
                if (term != null) {
                    dbHelper.lookup(term)
                } else {
                    LookupResult("none", null, emptyList())
                }
            }

            val totalTime = System.currentTimeMillis() - tStart

            _uiState.value = _uiState.value.copy(
                isSearching = false,
                extractedTerm = term,
                lookupResult = lookupRes,
                extractionTimeMs = tExtract,
                totalTimeMs = totalTime
            )

            if (isVoiceQuery && _uiState.value.autoSpeakVoiceResults) {
                val topDef = lookupRes.definitions.firstOrNull()
                if (topDef != null) {
                    textToSpeech.speakDefinition(
                        headword = topDef.headword,
                        pos = topDef.pos,
                        definition = topDef.definition,
                        example = topDef.example,
                        utteranceId = "voice_auto_${topDef.headword}_${topDef.pos}"
                    )
                } else if (lookupRes.closestMatches.isNotEmpty()) {
                    val suggestions = lookupRes.closestMatches.take(3).joinToString(", ")
                    textToSpeech.speak("No exact match found for ${lookupRes.resolvedHeadword ?: term}. Did you mean $suggestions?")
                } else if (term != null) {
                    textToSpeech.speak("No definition found for $term.")
                }
            }
        }
    }

    fun onVoiceResult(text: String) {
        val trimmed = text.trim()
        if (trimmed.isNotEmpty()) {
            _uiState.value = _uiState.value.copy(query = trimmed, isListening = false, errorMessage = null)
            performSearch(trimmed, isVoiceQuery = true)
        }
    }

    fun toggleVoiceListening(onFallbackLaunchIntent: ((android.content.Intent) -> Unit)? = null) {
        if (_uiState.value.isListening) {
            voiceRecognizer.stopListening()
            _uiState.value = _uiState.value.copy(isListening = false)
        } else {
            textToSpeech.stop()

            if (!android.speech.SpeechRecognizer.isRecognitionAvailable(getApplication())) {
                val intent = android.content.Intent(android.speech.RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                    putExtra(android.speech.RecognizerIntent.EXTRA_LANGUAGE_MODEL, android.speech.RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                    putExtra(android.speech.RecognizerIntent.EXTRA_LANGUAGE, "en-US")
                    putExtra(android.speech.RecognizerIntent.EXTRA_PROMPT, "Speak a word or question")
                }
                if (onFallbackLaunchIntent != null) {
                    onFallbackLaunchIntent(intent)
                } else {
                    _uiState.value = _uiState.value.copy(errorMessage = "Speech recognition is not available on this device.")
                }
                return
            }

            _uiState.value = _uiState.value.copy(isListening = true, errorMessage = null)
            voiceRecognizer.startListening(
                onListeningStateChanged = { listening ->
                    _uiState.value = _uiState.value.copy(isListening = listening)
                },
                onPartialResult = { partial ->
                    _uiState.value = _uiState.value.copy(query = partial)
                },
                onFinalResult = { final ->
                    onVoiceResult(final)
                },
                onError = { err ->
                    _uiState.value = _uiState.value.copy(isListening = false, errorMessage = err)
                }
            )
        }
    }

    override fun onCleared() {
        super.onCleared()
        voiceRecognizer.stopListening()
        textToSpeech.shutdown()
        extractor.close()
    }
}
