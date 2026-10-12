const menuButton = document.querySelector('.menu-toggle');
const nav = document.querySelector('.nav-links');
const toast = document.querySelector('.toast');
let toastTimer;

menuButton?.addEventListener('click', () => {
  const isOpen = menuButton.getAttribute('aria-expanded') === 'true';
  menuButton.setAttribute('aria-expanded', String(!isOpen));
  menuButton.setAttribute('aria-label', isOpen ? 'Open menu' : 'Close menu');
  nav?.classList.toggle('open', !isOpen);
});

nav?.querySelectorAll('a').forEach((link) => link.addEventListener('click', () => {
  menuButton?.setAttribute('aria-expanded', 'false');
  menuButton?.setAttribute('aria-label', 'Open menu');
  nav.classList.remove('open');
}));

document.querySelectorAll('[data-copy]').forEach((button) => button.addEventListener('click', async () => {
  const value = button.dataset.copy?.replaceAll('&quot;', '"');
  if (!value) return;
  try {
    await navigator.clipboard.writeText(value);
    toast.textContent = 'Command copied';
  } catch {
    toast.textContent = value;
  }
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 2200);
}));

document.querySelector('#year').textContent = new Date().getFullYear();
