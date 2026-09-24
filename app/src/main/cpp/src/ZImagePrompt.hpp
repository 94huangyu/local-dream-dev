// Exact reduction of Z-Image Turbo's Qwen chat template for Local Dream's
// initial text-to-image request: one user message, no tools, and
// enable_thinking=true. Do not use this for a multi-turn chat or tool call.
#ifndef ZIMAGEPROMPT_HPP
#define ZIMAGEPROMPT_HPP

#include <string>

inline std::string formatZImagePrompt(const std::string &prompt) {
  return "<|im_start|>user\n" + prompt +
         "<|im_end|>\n<|im_start|>assistant\n";
}

#endif  // ZIMAGEPROMPT_HPP
