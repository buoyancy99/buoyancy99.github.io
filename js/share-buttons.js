const shareBtns = document.querySelectorAll(".share-btn");

shareBtns.forEach((btn) => {
  btn.addEventListener("click", (e) => {
    e.preventDefault();
    let socialMedia = btn.className.split(" ")[1];
    let url = window.location.href; // Get current page URL
    let title = document.title; // Get current page title
    let img_url = document.querySelectorAll('meta[property="og:image"]')[0]
      .content;

    console.log(img_url); // Added line to print shareBtns

    switch (socialMedia) {
      case "twitter":
        window.open(
          `https://twitter.com/intent/tweet?text=Boyuan%20Chen%27s%20Homepage%3A%20${title}&url=${url}`
        );
        break;
      case "wechat":
        WeixinJSBridge.invoke("shareTimeline", {
          img_url: img_url,
          link: url,
          desc: `陈博远的主页：${title}`,
          title: `陈博远的主页：${title}`,
        });
        break;
      // case "linkedin":
      //   window.open(
      //     `https://www.linkedin.com/feed/?shareActive=true&text=Boyuan%20Chen%27s%20Homepage%3A+${title}+${url}`
      //   );
      //   break;
      // case "reddit":
      //   window.open(
      //     `https://www.reddit.com/submit?url=${url}&title=Boyuan+Chen+s+Homepage+${title}`
      //   );
      //   break;
    }
  });
});
