const ua = typeof navigator === "undefined" ? "" : navigator.userAgent;
const isMac = ua.includes("Macintosh") || ua.includes("Mac OS");

const CAP_SYMBOLS = {
  Command: isMac ? "⌘" : "Win",
  Cmd: isMac ? "⌘" : "Win",
  Super: isMac ? "⌘" : "Win",
  Meta: isMac ? "⌘" : "Win",
  MetaLeft: isMac ? "⌘" : "Win",
  MetaRight: isMac ? "⌘" : "Win",
  Control: isMac ? "⌃" : "Ctrl",
  Ctrl: isMac ? "⌃" : "Ctrl",
  ControlLeft: isMac ? "⌃" : "Ctrl",
  ControlRight: isMac ? "⌃" : "Ctrl",
  Alt: isMac ? "⌥" : "Alt",
  Option: isMac ? "⌥" : "Alt",
  AltLeft: isMac ? "⌥" : "Alt",
  AltRight: isMac ? "⌥" : "Alt",
  Shift: "⇧",
  ShiftLeft: "⇧",
  ShiftRight: "⇧",
};

const ARROW_SYMBOLS = {
  ArrowUp: "↑",
  ArrowDown: "↓",
  ArrowLeft: "←",
  ArrowRight: "→",
};

export function keycaps(accel) {
  if (!accel) return [];
  return accel.split("+").map((token) => {
    if (CAP_SYMBOLS[token]) return CAP_SYMBOLS[token];
    if (token.startsWith("Key")) return token.slice(3);
    if (token.startsWith("Digit")) return token.slice(5);
    if (ARROW_SYMBOLS[token]) return ARROW_SYMBOLS[token];
    return token;
  });
}
