document$.subscribe(() => {
  mermaid.initialize({
    startOnLoad: false,
  });
  mermaid.run();
});
