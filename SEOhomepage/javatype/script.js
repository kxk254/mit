document.addEventListener('DOMContentLoaded', () => {
    const navToggle = document.querySelector('.nav-toggle');
    const siteNav = document.querySelector('.site-nav');
    const navLinks = document.querySelectorAll('.site-nav a');

    // ハンバーガーメニューの開閉
    navToggle.addEventListener('click', () => {
        siteNav.classList.toggle('is-open');
        navToggle.classList.toggle('is-active');
        // aria-expanded属性の切り替え (アクセシビリティ向上)
        const isExpanded = navToggle.classList.contains('is-active');
        navToggle.setAttribute('aria-expanded', isExpanded);
    });

    // ナビゲーションリンククリック時にメニューを閉じる (モバイル時)
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            // CSSのtransition時間に合わせて少し遅延させるとスムーズ
            setTimeout(() => {
                siteNav.classList.remove('is-open');
                navToggle.classList.remove('is-active');
                navToggle.setAttribute('aria-expanded', 'false');
            }, 300); // CSSのtransition-durationと同じか少し長めに
        });
    });

    // オプション：スクロールで要素をフェードインさせる簡易的な例
    const fadeInElements = document.querySelectorAll('.service-item, .case-study-item, .about-section img');

    const fadeInObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = 1;
                entry.target.style.transform = 'translateY(0)';
                observer.unobserve(entry.target); // 一度表示したら監視を停止
            }
        });
    }, {
        rootMargin: '0px',
        threshold: 0.1 // 要素が10%見えたら発火
    });

    fadeInElements.forEach(el => {
        el.style.opacity = 0; // 初期状態は透明
        el.style.transform = 'translateY(20px)'; // 少し下に配置
        el.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out'; // アニメーション設定
        fadeInObserver.observe(el);
    });
});