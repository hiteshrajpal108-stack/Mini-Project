// Active TOC highlight on scroll
  const sections = document.querySelectorAll('section[id], h2[id], h3[id]');
  const tocLinks = document.querySelectorAll('.toc a');

  const tocEl = document.getElementById('toc-sidebar');
  const toolbarEl = document.getElementById('reader-toolbar');
  let isManualScroll = false;
  let manualScrollTimer = null;

  const setActiveTocLink = (link) => {
    if (!link) return;
    tocLinks.forEach(a => a.classList.remove('active'));
    link.classList.add('active');
    if (tocEl) {
      const linkTop = link.offsetTop;
      const linkBottom = linkTop + link.offsetHeight;
      if (linkTop < tocEl.scrollTop || linkBottom > tocEl.scrollTop + tocEl.clientHeight) {
        tocEl.scrollTo({ top: Math.max(0, linkTop - 40), behavior: 'smooth' });
      }
    }
  };

  const observer = new IntersectionObserver((entries) => {
    if (isManualScroll) return;
    const visible = entries.filter(e => e.isIntersecting);
    if (!visible.length) return;
    const best = visible[visible.length - 1];
    const link = document.querySelector(`.toc a[href="#${best.target.id}"]`);
    setActiveTocLink(link);
  }, { rootMargin: '-80px 0px -55% 0px', threshold: 0 });

  sections.forEach(s => observer.observe(s));

  tocLinks.forEach(a => a.addEventListener('click', (e) => {
    const href = a.getAttribute('href');
    if (!href || !href.startsWith('#')) return;
    const targetEl = document.getElementById(href.slice(1));
    if (!targetEl) return;

    e.preventDefault();
    isManualScroll = true;
    clearTimeout(manualScrollTimer);
    manualScrollTimer = setTimeout(() => { isManualScroll = false; }, 800);

    setActiveTocLink(a);

    const toolbarH = toolbarEl ? toolbarEl.getBoundingClientRect().height : 0;
    const curScroll = window.scrollY || window.pageYOffset || document.documentElement.scrollTop || 0;
    const targetY = targetEl.getBoundingClientRect().top + curScroll - toolbarH - 18;
    window.scrollTo({ top: Math.max(0, targetY), behavior: 'smooth' });
  }));

  // Smooth reading progress bar & back-to-top button
  const progressBar = document.getElementById('read-progress');
  const backToTop = document.getElementById('back-to-top');
  let ticking = false;

  const updateScrollUI = () => {
    const scrollTop = window.scrollY || document.documentElement.scrollTop;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    const pct = docHeight > 0 ? Math.min(100, Math.max(0, (scrollTop / docHeight) * 100)) : 0;
    if (progressBar) progressBar.style.width = pct.toFixed(2) + '%';
    if (backToTop) backToTop.classList.toggle('visible', scrollTop > 400);
    ticking = false;
  };

  window.addEventListener('scroll', () => {
    if (!ticking) {
      window.requestAnimationFrame(updateScrollUI);
      ticking = true;
    }
  }, { passive: true });

  if (backToTop) {
    backToTop.addEventListener('click', () => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  // Dynamic reading time estimate
  const articleEl = document.querySelector('article');
  const readTimePill = document.getElementById('read-time-pill');
  if (articleEl && readTimePill) {
    const words = (articleEl.innerText || '').trim().split(/\s+/).length;
    const mins = Math.max(1, Math.round(words / 220));
    readTimePill.textContent = `~${mins} min read · ${words.toLocaleString()} words`;
  }

  // Font size controls
  let fontSize = 15.5;
  const setFontSize = (sz) => {
    fontSize = Math.min(19.5, Math.max(13.5, sz));
    document.documentElement.style.setProperty('--article-size', fontSize + 'px');
  };
  document.getElementById('font-inc')?.addEventListener('click', () => setFontSize(fontSize + 1));
  document.getElementById('font-dec')?.addEventListener('click', () => setFontSize(fontSize - 1));

  // Light / Dark theme toggle
  const themeBtn = document.getElementById('theme-toggle');
  const applyTheme = (theme) => {
    if (theme === 'dark') {
      document.documentElement.setAttribute('data-theme', 'dark');
      if (themeBtn) themeBtn.textContent = '☀️ Light';
    } else {
      document.documentElement.removeAttribute('data-theme');
      if (themeBtn) themeBtn.textContent = '🌙 Dark';
    }
  };
  try {
    const savedTheme = localStorage.getItem('mhq_paper_theme');
    if (savedTheme === 'dark') applyTheme('dark');
  } catch (e) {}
  themeBtn?.addEventListener('click', () => {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const next = isDark ? 'light' : 'dark';
    applyTheme(next);
    try { localStorage.setItem('mhq_paper_theme', next); } catch (e) {}
  });

  // Print / PDF
  document.getElementById('print-paper')?.addEventListener('click', () => window.print());

  // Live in-paper search & highlight
  const searchInput = document.getElementById('paper-search');
  const searchCount = document.getElementById('search-count');
  let hits = [];
  let hitIdx = -1;

  const clearHighlights = () => {
    if (!articleEl) return;
    articleEl.querySelectorAll('mark.search-hit').forEach(mark => {
      const parent = mark.parentNode;
      parent.replaceChild(document.createTextNode(mark.textContent), mark);
      parent.normalize();
    });
    hits = [];
    hitIdx = -1;
  };

  const focusHit = (idx) => {
    if (!hits.length) {
      if (searchCount) searchCount.textContent = '0/0';
      return;
    }
    hits.forEach(h => h.classList.remove('current-hit'));
    hitIdx = ((idx % hits.length) + hits.length) % hits.length;
    const cur = hits[hitIdx];
    cur.classList.add('current-hit');
    const det = cur.closest('details');
    if (det) det.open = true;
    cur.scrollIntoView({ behavior: 'smooth', block: 'center' });
    if (searchCount) searchCount.textContent = `${hitIdx + 1}/${hits.length}`;
  };

  const runSearch = (query) => {
    clearHighlights();
    const q = (query || '').trim();
    if (!q || q.length < 2 || !articleEl) {
      if (searchCount) searchCount.textContent = '0/0';
      return;
    }
    const walker = document.createTreeWalker(articleEl, NodeFilter.SHOW_TEXT, {
      acceptNode: (node) => {
        if (!node.nodeValue || !node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
        const tag = node.parentElement?.tagName;
        if (tag === 'SCRIPT' || tag === 'STYLE' || tag === 'MARK') return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    const textNodes = [];
    while (walker.nextNode()) textNodes.push(walker.currentNode);
    const re = new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
    textNodes.forEach(node => {
      const val = node.nodeValue;
      let match;
      let lastIdx = 0;
      const frag = document.createDocumentFragment();
      let found = false;
      re.lastIndex = 0;
      while ((match = re.exec(val)) !== null) {
        found = true;
        if (match.index > lastIdx) frag.appendChild(document.createTextNode(val.slice(lastIdx, match.index)));
        const mark = document.createElement('mark');
        mark.className = 'search-hit';
        mark.textContent = match[0];
        frag.appendChild(mark);
        hits.push(mark);
        lastIdx = match.index + match[0].length;
      }
      if (found) {
        if (lastIdx < val.length) frag.appendChild(document.createTextNode(val.slice(lastIdx)));
        node.parentNode.replaceChild(frag, node);
      }
    });
    focusHit(0);
  };

  let searchTimer;
  searchInput?.addEventListener('input', (e) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => runSearch(e.target.value), 160);
  });
  searchInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      focusHit(e.shiftKey ? hitIdx - 1 : hitIdx + 1);
    }
  });
  document.getElementById('search-next')?.addEventListener('click', () => focusHit(hitIdx + 1));
  document.getElementById('search-prev')?.addEventListener('click', () => focusHit(hitIdx - 1));

  // Click-to-zoom figure lightbox
  const lightbox = document.getElementById('fig-lightbox');
  const lbImg = document.getElementById('fig-lightbox-img');
  const lbCap = document.getElementById('fig-lightbox-cap');
  document.querySelectorAll('figure').forEach(fig => {
    const img = fig.querySelector('img');
    const cap = fig.querySelector('figcaption');
    if (img) {
      img.addEventListener('click', () => {
        if (lbImg) { lbImg.src = img.src; lbImg.alt = img.alt; }
        if (lbCap) lbCap.innerHTML = cap ? cap.innerHTML : img.alt;
        lightbox?.classList.add('open');
      });
    }
  });
  const closeLightbox = () => lightbox?.classList.remove('open');
  document.getElementById('fig-lightbox-close')?.addEventListener('click', closeLightbox);
  lightbox?.addEventListener('click', (e) => { if (e.target === lightbox) closeLightbox(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeLightbox(); });
